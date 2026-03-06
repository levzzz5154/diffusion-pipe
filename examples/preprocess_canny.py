#!/usr/bin/env python3
"""
Preprocess images to extract Canny edge maps for control LoRA training.

Usage:
    python examples/preprocess_canny.py /path/to/images /path/to/canny_output

Each image in the source directory will have its Canny edges extracted and saved
as a 3-channel RGB image (required for VAE encoding) in the output directory
with the same filename stem.
"""

import argparse
import sys
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}


def extract_canny(image_path, low_threshold=100, high_threshold=200):
    """Extract Canny edges from an image and return as 3-channel RGB."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Could not read image: {image_path}')
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def main():
    parser = argparse.ArgumentParser(description='Extract Canny edge maps from images.')
    parser.add_argument('input_dir', type=Path, help='Directory containing source images')
    parser.add_argument('output_dir', type=Path, help='Directory to save Canny edge maps')
    parser.add_argument('--low', type=int, default=100, help='Canny low threshold (default: 100)')
    parser.add_argument('--high', type=int, default=200, help='Canny high threshold (default: 200)')
    parser.add_argument('--ext', type=str, default='.png', help='Output file extension (default: .png)')
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f'Error: {args.input_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f'No images found in {args.input_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Processing {len(image_files)} images...')
    for img_path in image_files:
        edges_rgb = extract_canny(img_path, args.low, args.high)
        out_path = args.output_dir / (img_path.stem + args.ext)
        cv2.imwrite(str(out_path), edges_rgb)
        print(f'  {img_path.name} -> {out_path.name}')

    print(f'Done. Canny edge maps saved to {args.output_dir}')


if __name__ == '__main__':
    main()
