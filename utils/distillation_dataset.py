"""Dataset for distillation training using pre-saved teacher outputs.

Loads .safetensors files containing pre-computed latents and noise from a
teacher model, along with prompts stored in the file metadata. Each file
represents one training sample.

Expected safetensors contents:
  - "noise"  tensor: starting noise  (shape [1, C, 1, H, W] or [1, C, H, W])
  - "latent" tensor: clean latent     (same shape convention)
  - metadata keys: "positive_prompt", "negative_prompt"
"""

import os
import random
from pathlib import Path
from collections import defaultdict

import torch
import datasets
from deepspeed.utils.logging import logger

from safetensors.torch import load_file
from safetensors import safe_open

from utils.dataset import (
    Dataset,
    _cache_text_embeddings,
    _map_and_cache,
    UNCOND_FRACTION,
)
from utils.common import is_main_process


class DistillationSizeBucketDataset:
    """Leaf dataset holding pre-saved distillation samples for a single bucket.

    Mirrors the interface of ``SizeBucketDataset`` so that
    ``ConcatenatedBatchedDataset`` and the rest of the pipeline work unchanged.
    """

    def __init__(self, latents_list, noise_list, captions, image_specs,
                 metadata_dataset, size_bucket, directory_config, cache_dir):
        self.latents_list = latents_list
        self.noise_list = noise_list
        self.captions = captions
        self.image_specs = image_specs
        self.metadata_dataset = metadata_dataset
        self.size_bucket = tuple(int(x) for x in size_bucket)
        self.directory_config = directory_config
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.text_embedding_datasets = []
        self.uncond_text_embeddings = []
        self.num_repeats = directory_config.get('num_repeats', 1)
        if self.num_repeats <= 0:
            raise ValueError(f'num_repeats must be >0, was {self.num_repeats}')

        self.iteration_order = datasets.Dataset.from_dict({
            'image_spec': image_specs,
            'latents_idx': list(range(len(captions))),
            'caption': captions,
            'caption_number': [0] * len(captions),
        })

    # ------------------------------------------------------------------
    # Caching interface
    # ------------------------------------------------------------------

    def cache_latents(self, map_fn, regenerate_cache=False, trust_cache=False, caching_batch_size=1):
        # No-op: latents are pre-computed and already in memory.
        pass

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        te_dataset = _cache_text_embeddings(
            self.metadata_dataset, map_fn, i, self.cache_dir,
            regenerate_cache, caching_batch_size,
        )
        self.text_embedding_datasets.append(te_dataset)

    def add_text_embedding_dataset(self, te_dataset):
        self.text_embedding_datasets.append(te_dataset)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        idx = idx % len(self.iteration_order)
        entry = self.iteration_order[idx]

        latents_idx = entry['latents_idx']
        ret = {
            'latents': self.latents_list[latents_idx],
            'noise': self.noise_list[latents_idx],
            'mask': None,
        }

        use_uncond = UNCOND_FRACTION > 0 and random.random() < UNCOND_FRACTION
        caption = '' if use_uncond else entry['caption']

        for ds, uncond_ds in zip(self.text_embedding_datasets, self.uncond_text_embeddings):
            emb_dict = uncond_ds[0] if use_uncond else ds.get_text_embeddings(
                tuple(entry['image_spec']), entry['caption_number'],
            )
            ret.update(emb_dict)

        ret['caption'] = caption
        return ret

    def __len__(self):
        return int(len(self.iteration_order) * self.num_repeats)


class DistillationDirectoryDataset:
    """Scans a directory of ``.safetensors`` distillation files.

    Mirrors the ``DirectoryDataset`` interface expected by the outermost
    ``Dataset`` / ``DatasetManager`` classes.
    """

    def __init__(self, directory_config, dataset_config, model_name):
        directory_config.setdefault('num_repeats', dataset_config.get('num_repeats', 1))
        self.directory_config = directory_config
        self.dataset_config = dataset_config
        self.path = Path(directory_config['path'])
        self.model_name = model_name
        self.cache_dir = self.path / 'cache' / model_name
        self.size_bucket_datasets = []

        if not self.path.exists() or not self.path.is_dir():
            raise RuntimeError(f'Invalid distillation data path: {self.path}')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_safetensors_files(self):
        """Load all ``.safetensors`` files and group by latent spatial size."""
        files = sorted(self.path.glob('*.safetensors'))
        if len(files) == 0:
            raise RuntimeError(f'No .safetensors files found in {self.path}')

        # WanVAE channel-wise normalization (process_in).  ComfyUI's sampler
        # applies process_out to denoised latents before returning them, so
        # saved latents are in denormalized space.  diffusion-pipe trains in
        # normalized space, so we must apply process_in here.
        wan_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, 16, 1, 1, 1)
        wan_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, 16, 1, 1, 1)

        # Group samples by (width, height, frames) so different resolutions
        # end up in separate size buckets.
        buckets = defaultdict(lambda: {
            'latents': [], 'noise': [], 'captions': [], 'image_specs': [],
        })

        for f in files:
            data = load_file(str(f))
            with safe_open(str(f), framework="pt") as sf:
                metadata = sf.metadata()

            latent = data['latent'].float()
            noise = data['noise'].float()

            # Normalise to 5-D [1, C, F, H, W] then drop the batch dim -> [C, F, H, W]
            if latent.dim() == 4:
                latent = latent.unsqueeze(2)
            if noise.dim() == 4:
                noise = noise.unsqueeze(2)

            # Apply WanVAE process_in: saved latent is in denormalized space
            # (ComfyUI applies process_out after sampling), but diffusion-pipe
            # trains in normalized space.  Noise is already raw N(0,1).
            latent = (latent - wan_mean) / wan_std

            latent = latent.squeeze(0)
            noise = noise.squeeze(0)

            caption = metadata.get('positive_prompt', '') if metadata else ''

            _, frames, h, w = latent.shape
            size_bucket = (int(w), int(h), int(frames))
            bucket = buckets[size_bucket]
            bucket['latents'].append(latent)
            bucket['noise'].append(noise)
            bucket['captions'].append(caption)
            bucket['image_specs'].append((None, str(f)))

        if is_main_process():
            print(f'Loaded {len(files)} distillation samples from {self.path}')
            for sb, b in buckets.items():
                print(f'  Latent size bucket {sb}: {len(b["latents"])} samples')

        return buckets

    # ------------------------------------------------------------------
    # Caching interface (called by DatasetManager / _cache_fn)
    # ------------------------------------------------------------------

    def cache_metadata(self, regenerate_cache=False, trust_cache=False):
        buckets = self._load_safetensors_files()

        metadata_dir = self.cache_dir / 'metadata'
        os.makedirs(metadata_dir, exist_ok=True)

        self.size_bucket_datasets = []
        for size_bucket, bucket_data in buckets.items():
            # Build HF Dataset for text-embedding caching
            metadata_save_path = metadata_dir / f'distillation_metadata_{size_bucket[0]}x{size_bucket[1]}x{size_bucket[2]}'
            if not metadata_save_path.exists() or regenerate_cache or not trust_cache:
                md = datasets.Dataset.from_dict({
                    'image_spec': bucket_data['image_specs'],
                    'caption': [[c] for c in bucket_data['captions']],
                    'is_video': [False] * len(bucket_data['captions']),
                })
                md.save_to_disk(str(metadata_save_path))
            metadata_dataset = datasets.load_from_disk(str(metadata_save_path))

            bucket_cache_dir = self.cache_dir / f'cache_{size_bucket[0]}x{size_bucket[1]}x{size_bucket[2]}'
            self.size_bucket_datasets.append(
                DistillationSizeBucketDataset(
                    bucket_data['latents'], bucket_data['noise'],
                    bucket_data['captions'], bucket_data['image_specs'],
                    metadata_dataset, size_bucket, self.directory_config,
                    bucket_cache_dir,
                )
            )

    def get_size_bucket_datasets(self):
        return self.size_bucket_datasets

    def cache_latents(self, map_fn, regenerate_cache=False, trust_cache=False, caching_batch_size=1):
        for ds in self.size_bucket_datasets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache,
                             trust_cache=trust_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        for ds in self.size_bucket_datasets:
            ds.cache_text_embeddings(map_fn, i, regenerate_cache=regenerate_cache,
                                     caching_batch_size=caching_batch_size)
        # Cache unconditional (empty-prompt) text embeddings, matching
        # the behaviour of DirectoryDataset.
        empty_caption_ds = datasets.Dataset.from_dict({
            'caption': [''], 'is_video': [False], 'image_spec': [(None, None)],
        })
        uncond_text_embeddings_ds = _map_and_cache(
            empty_caption_ds, map_fn, cache_dir=self.cache_dir,
            cache_file_prefix=f'uncond_text_embeddings_{i}_',
            regenerate_cache=regenerate_cache,
        )
        for ds in self.size_bucket_datasets:
            ds.uncond_text_embeddings.append(uncond_text_embeddings_ds)


class DistillationDataset(Dataset):
    """Top-level dataset for distillation training.

    Drop-in replacement for ``Dataset`` (from ``utils.dataset``) that reads
    pre-saved teacher outputs instead of raw media files.  Register it with
    ``DatasetManager`` exactly like a normal ``Dataset``::

        distill_data = DistillationDataset(dataset_config, model)
        dataset_manager.register(distill_data)
        dataset_manager.cache()
        distill_data.post_init(...)
    """

    def __init__(self, dataset_config, model, skip_dataset_validation=False):
        # Intentionally bypass Dataset.__init__ to substitute
        # DistillationDirectoryDataset for DirectoryDataset.
        self.dataset_config = dataset_config
        self.model = model
        self.model_name = model.name
        self.post_init_called = False
        self.eval_quantile = None

        self.directory_datasets = []
        for directory_config in dataset_config['directory']:
            directory_dataset = DistillationDirectoryDataset(
                directory_config, dataset_config, self.model_name,
            )
            self.directory_datasets.append(directory_dataset)
