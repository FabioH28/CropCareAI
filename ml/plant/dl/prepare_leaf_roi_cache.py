"""Precompute leaf-focused training images for field-first training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.classifier.data import CropDiseaseDataset, load_manifest
from ml.plant.dl.classifier.paths import ROI_CACHE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute leaf ROI cache files for plant model training.")
    parser.add_argument("--input-mode", default="leaf_isolated", choices=["leaf_context", "leaf_isolated"])
    parser.add_argument("--cache-dir", default=None, help="Optional custom cache directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to process for a smoke run.")
    parser.add_argument("--source", default=None, choices=["plantvillage", "plantdoc", "mendeley_tomato"], help="Optional source filter.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    if args.source:
        manifest = manifest[manifest["source"] == args.source].copy()
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    cache_dir = Path(args.cache_dir) if args.cache_dir else ROI_CACHE_ROOT / args.input_mode
    dataset = CropDiseaseDataset(
        manifest,
        class_to_index={label: idx for idx, label in enumerate(sorted(manifest["canonical_label"].unique().tolist()))},
        transform=None,
        input_mode=args.input_mode,
        roi_cache_root=cache_dir,
    )

    for index in tqdm(range(len(dataset)), desc=f"caching_{args.input_mode}"):
        dataset[index]

    print({"cache_dir": str(cache_dir), "items_processed": len(dataset)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
