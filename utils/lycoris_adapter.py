from lycoris.modules.loha import LohaModule as _LohaModule
from lycoris.modules.lokr import LokrModule as _LokrModule

import torch
from torch import nn

LYCORIS_MODULE_CLASSES = {
    'loha': _LohaModule,
    'lokr': _LokrModule,
}

LYCORIS_WEIGHT_KEYS = {
    'loha': ['hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b', 'hada_t1', 'hada_t2', 'alpha', 'dora_scale'],
    'lokr': ['lokr_w1', 'lokr_w1_a', 'lokr_w1_b', 'lokr_w2', 'lokr_w2_a', 'lokr_w2_b', 'lokr_t1', 'lokr_t2', 'alpha', 'dora_scale'],
}


def configure_lycoris(transformer, adapter_target_modules, adapter_config):
    module_cls = LYCORIS_MODULE_CLASSES[adapter_config['type']]
    algo = adapter_config['type']
    rank = adapter_config['rank']
    alpha = adapter_config['alpha']
    dropout = adapter_config.get('dropout', 0.0)
    factor = adapter_config.get('factor', -1)

    targets = []
    for name, module in transformer.named_modules():
        if module.__class__.__name__ not in adapter_target_modules:
            continue
        for full_submodule_name, submodule in module.named_modules(prefix=name):
            if isinstance(submodule, nn.Linear):
                targets.append((full_submodule_name, submodule))

    lycoris_modules = []
    for full_submodule_name, submodule in targets:
        parts = full_submodule_name.split(".")
        child_name = parts[-1]
        parent_path = ".".join(parts[:-1])
        parent = transformer.get_submodule(parent_path)

        lora_name = full_submodule_name.replace(".", "_")

        kwargs = dict(
            lora_name=lora_name,
            org_module=submodule,
            multiplier=1.0,
            lora_dim=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if algo == 'lokr':
            kwargs['factor'] = factor

        lycoris_mod = module_cls(**kwargs)
        lycoris_mod.apply_to()

        register_parent = parent
        suffix_parts = [child_name]
        current_path = parent_path
        while isinstance(register_parent, nn.Sequential):
            if current_path:
                suffix_parts.insert(0, current_path.split(".")[-1])
                current_path = ".".join(current_path.split(".")[:-1])
                register_parent = transformer.get_submodule(current_path) if current_path else transformer
            else:
                register_parent = transformer
                break
        register_name = "_lycoris_" + "_".join(suffix_parts)
        register_parent.add_module(register_name, lycoris_mod)

        for param_name, p in lycoris_mod.named_parameters():
            p.original_name = f"{full_submodule_name}.{param_name}"

        lycoris_modules.append(lycoris_mod)

    return lycoris_modules


def build_lycoris_name_map(transformer):
    name_map = {}
    for name, p in transformer.named_parameters():
        if hasattr(p, 'original_name'):
            name_map[p.original_name] = name
    return name_map


def is_lycoris_state_dict_key(key):
    key_after_prefix = key.split('.', 1)[1] if '.' in key else key
    for suffixes in LYCORIS_WEIGHT_KEYS.values():
        for suffix in suffixes:
            if key_after_prefix.endswith(suffix) or f'.{suffix}' in key:
                return True
    return False
