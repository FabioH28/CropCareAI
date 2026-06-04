"""Train the CropCare AI disease classifier."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.classifier.trainer import TrainConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CropCare AI disease classifier.")
    parser.add_argument("--model-name", default="tf_efficientnetv2_s")
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-stage1", type=int, default=8)
    parser.add_argument("--epochs-stage2", type=int, default=5)
    parser.add_argument("--learning-rate-stage1", type=float, default=3e-4)
    parser.add_argument("--learning-rate-stage2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--output-name", default="best_model")
    parser.add_argument("--promote-to-production", action="store_true")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--input-mode", default="full_image", choices=["full_image", "leaf_context", "leaf_isolated"])
    parser.add_argument("--augmentation-profile", default="standard", choices=["standard", "field_heavy"])
    parser.add_argument("--roi-cache-dir", default=None)
    parser.add_argument("--field-source-boost", type=float, default=1.0)
    parser.add_argument("--stage2-controlled-fraction", type=float, default=0.20)
    parser.add_argument("--loss-name", default="cross_entropy", choices=["cross_entropy", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--target-column", default="canonical_label", choices=["canonical_label", "crop"], help="Train a disease classifier (canonical_label) or a crop classifier (crop).")
    parser.add_argument("--crop-filter", default=None, help="Optional crop name to train a per-crop disease specialist (e.g., 'apple').")
    parser.add_argument("--field-val-size", type=float, default=0.0, help="Fraction of field (PlantDoc) train images to hold out as a field-validation split for stage-2 checkpoint selection. 0 disables (default).")
    parser.add_argument("--select-on-field", action="store_true", help="Select the stage-2 checkpoint by field-val macro-F1 instead of controlled-val. Requires --field-val-size > 0.")
    parser.add_argument("--extra-field-source", action="append", default=[], help="Extra manifest source name to add to stage-2 field training (e.g. 'plant_pathology_2021'). May be repeated. Added after the field-val carve so validation stays on the PlantDoc distribution.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        learning_rate_stage1=args.learning_rate_stage1,
        learning_rate_stage2=args.learning_rate_stage2,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
        output_name=args.output_name,
        promote_to_production=args.promote_to_production,
        grad_accumulation_steps=args.grad_accumulation_steps,
        amp_enabled=not args.disable_amp,
        early_stopping_patience=args.early_stopping_patience,
        input_mode=args.input_mode,
        augmentation_profile=args.augmentation_profile,
        roi_cache_dir=args.roi_cache_dir,
        field_source_boost=args.field_source_boost,
        stage2_controlled_fraction=args.stage2_controlled_fraction,
        loss_name=args.loss_name,
        focal_gamma=args.focal_gamma,
        target_column=args.target_column,
        crop_filter=args.crop_filter,
        field_val_size=args.field_val_size,
        select_stage2_on_field=args.select_on_field,
        extra_field_sources=tuple(args.extra_field_source),
    )
    summary = run_training(config)
    print(json.dumps(summary, indent=2))
