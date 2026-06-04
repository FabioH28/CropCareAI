"""Cross-crop PlantDoc field evaluation of the live cascade.

Reproduces the headline field numbers (per-crop and overall) by running the
deployed cascade — unified crop model -> per-crop specialist -> weighted
geometric-mean ensemble -> open-set gate — on the PlantDoc *test* split,
restricted to the six supported crops.

Methodology (identical to the original cross_crop_cascade_eval.json):
  - one prediction per image via cascade_predict
  - correct == predicted canonical_label equals the true canonical_label
  - thresholds and TTA match the live backend defaults (0.65 / 0.50 / tta=1)

Usage:
    python scripts/eval_cross_crop.py
    python scripts/eval_cross_crop.py --output ml/plant/artifacts/reports/cross_crop_cascade_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.cascade import cascade_predict
from ml.plant.dl.classifier.data import load_manifest

SUPPORTED_CROPS = ["apple", "tomato", "grape", "corn", "pepper", "potato"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-crop PlantDoc field evaluation of the live cascade.")
    parser.add_argument("--unified", default="ml/plant/artifacts/production/best_model.pt")
    parser.add_argument("--checkpoint-dir", default="ml/plant/artifacts/checkpoints")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--routing-threshold", type=float, default=0.50)
    parser.add_argument("--tta-passes", type=int, default=4)  # matches the live backend (.env DL_TTA_PASSES=4)
    parser.add_argument("--output", default="ml/plant/artifacts/reports/cross_crop_cascade_eval_v2.json")
    args = parser.parse_args()

    unified = (PROJECT_ROOT / args.unified).resolve()
    checkpoint_dir = (PROJECT_ROOT / args.checkpoint_dir).resolve()
    specialists = {
        crop: path
        for crop in SUPPORTED_CROPS
        if (path := checkpoint_dir / f"{crop}_specialist.pt").exists()
    }

    manifest = load_manifest()
    field = manifest[
        (manifest["source"] == "plantdoc")
        & (manifest["source_split"] == "test")
        & (manifest["crop"].isin(SUPPORTED_CROPS))
    ].dropna(subset=["canonical_label", "path"]).reset_index(drop=True)

    per_crop_total: Counter = Counter()
    per_crop_correct: Counter = Counter()
    route_counts: Counter = Counter()
    per_image = []
    correct_total = 0

    for i, row in field.iterrows():
        true_label = row["canonical_label"]
        crop = row["crop"]
        result = cascade_predict(
            Path(row["path"]),
            unified,
            specialists,
            threshold=args.threshold,
            routing_threshold=args.routing_threshold,
            tta_passes=args.tta_passes,
        )
        predicted = result.get("predicted_label")
        is_correct = bool(predicted == true_label)
        per_crop_total[crop] += 1
        per_crop_correct[crop] += int(is_correct)
        route_counts[result.get("cascade_route", "unknown")] += 1
        correct_total += int(is_correct)
        per_image.append(
            {
                "image": row["path"],
                "true_label": true_label,
                "predicted_label": predicted,
                "confidence": round(float(result.get("confidence") or 0.0), 4),
                "cascade_used": bool(result.get("cascade_used")),
                "cascade_route": result.get("cascade_route"),
                "correct": is_correct,
            }
        )
        print(f"[{i + 1}/{len(field)}] {crop:7} true={true_label:28} pred={predicted}  {'ok' if is_correct else 'X'}", flush=True)

    total = len(field)
    summary = {
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / max(total, 1),
        "per_crop_total": dict(per_crop_total),
        "per_crop_correct": dict(per_crop_correct),
        "route_counts": dict(route_counts),
        "specialists_registered": sorted(specialists),
        "settings": {
            "threshold": args.threshold,
            "routing_threshold": args.routing_threshold,
            "tta_passes": args.tta_passes,
            "unified": str(unified),
        },
        "per_image": per_image,
    }

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== cross-crop field accuracy (live cascade) ===")
    print(f"{'crop':8} {'n':>4} {'accuracy':>10}")
    for crop in SUPPORTED_CROPS:
        n = per_crop_total.get(crop, 0)
        if n:
            print(f"{crop:8} {n:>4} {per_crop_correct[crop] / n * 100:>9.1f}%")
    print(f"{'OVERALL':8} {total:>4} {summary['accuracy'] * 100:>9.1f}%")
    print(f"\nwrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
