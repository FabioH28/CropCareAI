"""Hierarchical cascade inference: unified model -> per-crop specialist.

Strategy:
  1. Run the unified model on the image.
  2. If it predicts a crop with an available specialist AND its confidence is
     above the routing threshold, run the specialist on the same image.
  3. Use the specialist's class label as the final prediction; keep the
     unified model's crop-detection trail in the result.

Why: the specialist sees only one crop's data during training, so it can
learn finer disease distinctions without competing with the other classes
for network capacity. On apple specifically this raises PlantDoc accuracy
~5 points over the unified baseline.

Falls back to the unified prediction when:
  - the predicted crop has no specialist registered
  - the unified is below the routing threshold (likely wrong crop guess)
  - the open-set gate marked the input as outside the supported crop set
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.classifier.label_schema import CANONICAL_LABELS
from ml.plant.dl.infer import predict


# Weighted geometric mean of specialist(s) + unified-restricted-to-specialist-labels.
# Total specialist weight 0.7 (split equally if multiple specialists),
# unified weight 0.3. Specialist evidence dominates because each specialist
# was trained on this crop's data only.
DEFAULT_SPECIALIST_TOTAL_WEIGHT = 0.7
DEFAULT_UNIFIED_WEIGHT = 0.3
DEFAULT_SPECIALIST_WEIGHT = DEFAULT_SPECIALIST_TOTAL_WEIGHT  # back-compat alias


def _restrict_and_renormalize(
    full_probs: dict[str, float] | None,
    target_labels: list[str],
) -> dict[str, float]:
    """Filter the unified's softmax to the specialist's label set and rescale.

    The unified model outputs a distribution over ~25 labels. The specialist
    only knows 4 apple labels. To ensemble them we restrict the unified
    distribution to those same 4 labels and re-normalize so they sum to 1 —
    this is the unified's "if forced to pick among apple classes" view.
    """
    if not full_probs:
        return {label: 1.0 / max(len(target_labels), 1) for label in target_labels}
    restricted = {label: max(float(full_probs.get(label, 0.0)), 0.0) for label in target_labels}
    total = sum(restricted.values())
    if total <= 0:
        return {label: 1.0 / max(len(target_labels), 1) for label in target_labels}
    return {label: value / total for label, value in restricted.items()}


def _weighted_geometric_mean(
    specialist_probs: dict[str, float],
    unified_restricted_probs: dict[str, float],
    *,
    specialist_weight: float = DEFAULT_SPECIALIST_WEIGHT,
    unified_weight: float = DEFAULT_UNIFIED_WEIGHT,
    epsilon: float = 1e-9,
) -> dict[str, float]:
    """Backward-compatible two-distribution ensemble."""
    return _weighted_geometric_mean_many(
        [(specialist_probs, specialist_weight), (unified_restricted_probs, unified_weight)],
        epsilon=epsilon,
    )


def _weighted_geometric_mean_many(
    weighted_distributions: list[tuple[dict[str, float], float]],
    *,
    epsilon: float = 1e-9,
) -> dict[str, float]:
    """Generalized weighted geometric mean over any number of distributions.

    Each item is (probability_dict, weight). The result is normalized with
    softmax over the log-scores so it is a proper probability distribution
    over the union of all labels seen.
    """
    if not weighted_distributions:
        return {}
    label_set: set[str] = set()
    for probs, _ in weighted_distributions:
        label_set.update(probs.keys())
    labels = sorted(label_set)
    log_scores: dict[str, float] = {}
    for label in labels:
        score = 0.0
        for probs, weight in weighted_distributions:
            p = max(float(probs.get(label, 0.0)), epsilon)
            score += weight * math.log(p)
        log_scores[label] = score
    max_log = max(log_scores.values())
    exps = {label: math.exp(score - max_log) for label, score in log_scores.items()}
    denom = sum(exps.values()) or 1.0
    return {label: value / denom for label, value in exps.items()}


def cascade_predict(
    image_path: Path,
    unified_checkpoint: Path,
    specialists_by_crop: dict[str, Path | list[Path]],
    *,
    threshold: float = 0.65,
    routing_threshold: float = 0.50,
    tta_passes: int = 1,
) -> dict[str, Any]:
    """Run the cascade with one or more specialists per crop.

    Args:
        image_path: input image.
        unified_checkpoint: the multi-class unified disease model.
        specialists_by_crop: map from crop name to either a single specialist
            checkpoint path OR a list of paths (multi-checkpoint ensemble).
            Multiple checkpoints for the same crop get equal weight within
            the specialist contribution; together they contribute
            DEFAULT_SPECIALIST_TOTAL_WEIGHT and the unified model contributes
            DEFAULT_UNIFIED_WEIGHT.
        threshold: confidence threshold below which a class becomes
            "uncertain" inside ``predict``.
        routing_threshold: minimum unified confidence required before
            routing to a specialist.
        tta_passes: number of test-time augmentation passes (1-4).

    Returns the unified result with cascade fields appended.
    """
    unified_result = predict(image_path, unified_checkpoint, threshold, tta_passes=tta_passes)
    raw_crop = unified_result.get("matched_supported_crop") or unified_result.get("crop")
    confidence = float(unified_result.get("confidence") or 0.0)

    cascade_route = "unified_only"
    final = dict(unified_result)
    final["cascade_used"] = False
    final["unified_prediction"] = {
        "predicted_label": unified_result.get("predicted_label"),
        "crop": unified_result.get("crop"),
        "matched_supported_crop": raw_crop,
        "disease": unified_result.get("disease"),
        "confidence": confidence,
    }

    if raw_crop not in specialists_by_crop:
        cascade_route = "no_specialist_for_crop"
    elif confidence < routing_threshold:
        cascade_route = f"low_confidence_below_{routing_threshold:.2f}"
    else:
        # Normalize to a list so single-path and multi-path callers share code.
        raw_specs = specialists_by_crop[raw_crop]
        specialist_paths_raw = raw_specs if isinstance(raw_specs, (list, tuple)) else [raw_specs]
        specialist_paths = [Path(p) for p in specialist_paths_raw if Path(p).exists()]

        if not specialist_paths:
            cascade_route = "specialist_checkpoint_missing"
        else:
            # Run every registered specialist for this crop.
            specialist_results = []
            for spec_path in specialist_paths:
                res = predict(image_path, spec_path, threshold, tta_passes=tta_passes)
                res["__checkpoint__"] = str(spec_path)
                specialist_results.append(res)

            # Use the first specialist's labels as the reference label set
            # (all specialists for one crop should share the same schema).
            specialist_labels = sorted((specialist_results[0].get("class_probabilities") or {}).keys())

            unified_restricted = _restrict_and_renormalize(
                unified_result.get("class_probabilities"),
                specialist_labels,
            )

            # Weighted geometric mean across all specialist softmaxes (equal
            # within the specialist pool) plus the unified-restricted view.
            n_specs = len(specialist_results)
            per_spec_weight = DEFAULT_SPECIALIST_TOTAL_WEIGHT / n_specs
            weighted_distributions = [
                (res.get("class_probabilities") or {}, per_spec_weight)
                for res in specialist_results
            ]
            weighted_distributions.append((unified_restricted, DEFAULT_UNIFIED_WEIGHT))
            ensemble_probs = _weighted_geometric_mean_many(weighted_distributions)

            ensemble_label = max(ensemble_probs, key=ensemble_probs.get)
            ensemble_confidence = float(ensemble_probs[ensemble_label])
            ensemble_info = CANONICAL_LABELS.get(ensemble_label)

            # The "headline" specialist for the backward-compatible fields is
            # the most-confident individual specialist.
            best_spec_result = max(
                specialist_results,
                key=lambda r: float(r.get("confidence") or 0.0),
            )
            specialist_label = best_spec_result.get("matched_supported_label") or best_spec_result["predicted_label"]
            specialist_info = CANONICAL_LABELS.get(specialist_label)
            specialist_confidence = float(best_spec_result["confidence"])

            final["predicted_label"] = ensemble_label
            if ensemble_info is not None:
                final["crop"] = ensemble_info.crop
                final["disease"] = ensemble_info.disease
                final["is_healthy"] = ensemble_info.is_healthy
                final["health_status"] = "healthy" if ensemble_info.is_healthy else (
                    "uncertain" if ensemble_confidence < threshold else "diseased"
                )
            final["confidence"] = ensemble_confidence
            final["top_predictions"] = [
                {"label": label, "probability": ensemble_probs[label]}
                for label in sorted(ensemble_probs, key=ensemble_probs.get, reverse=True)[:3]
            ]
            # Downstream branch fusion uses class_probabilities — give it the
            # ensembled distribution rather than the specialist-only one.
            final["unified_class_probabilities"] = unified_result.get("class_probabilities")
            final["class_probabilities"] = ensemble_probs
            final["cascade_used"] = True
            final["specialist_prediction"] = {
                "predicted_label": specialist_label,
                "crop": specialist_info.crop if specialist_info else None,
                "disease": specialist_info.disease if specialist_info else None,
                "confidence": specialist_confidence,
                "class_probabilities": best_spec_result.get("class_probabilities"),
                "checkpoint": best_spec_result.get("__checkpoint__"),
            }
            final["specialist_predictions"] = [
                {
                    "checkpoint": res.get("__checkpoint__"),
                    "predicted_label": res.get("matched_supported_label") or res.get("predicted_label"),
                    "confidence": float(res.get("confidence") or 0.0),
                    "class_probabilities": res.get("class_probabilities"),
                }
                for res in specialist_results
            ]
            final["ensemble_prediction"] = {
                "predicted_label": ensemble_label,
                "crop": ensemble_info.crop if ensemble_info else None,
                "disease": ensemble_info.disease if ensemble_info else None,
                "confidence": ensemble_confidence,
                "specialist_total_weight": DEFAULT_SPECIALIST_TOTAL_WEIGHT,
                "per_specialist_weight": per_spec_weight,
                "unified_weight": DEFAULT_UNIFIED_WEIGHT,
                "specialist_count": n_specs,
                "unified_restricted_probs": unified_restricted,
                "method": "weighted_geometric_mean_multi_specialist_plus_unified",
            }
            # The unified open-set gate was built on pre-PP2020/PP2021
            # embeddings and rejects legitimate field photos. When the
            # ensemble confidently agrees on the crop, trust it over the
            # stale gate.
            if ensemble_confidence >= routing_threshold:
                final["is_known_crop"] = True
                final["matched_supported_label"] = ensemble_label
                final["matched_supported_crop"] = ensemble_info.crop if ensemble_info else final.get("matched_supported_crop")
            route_suffix = "specialists_plus_unified" if n_specs > 1 else "specialist_plus_unified"
            cascade_route = f"ensemble_{raw_crop}_{n_specs}_{route_suffix}"

    final["cascade_route"] = cascade_route
    return final


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Run hierarchical cascade inference.")
    parser.add_argument("image_path")
    parser.add_argument("--unified", default="ml/plant/artifacts/production/best_model.pt")
    parser.add_argument(
        "--specialist",
        action="append",
        default=[],
        help="Per-crop specialist in the form 'crop=path/to/checkpoint.pt'. May be repeated.",
    )
    parser.add_argument(
        "--apple-specialist",
        default=None,
        help="Backward-compatible shortcut for --specialist apple=PATH.",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--routing-threshold", type=float, default=0.50)
    parser.add_argument("--tta-passes", type=int, default=1)
    args = parser.parse_args()

    specialists: dict[str, list[Path]] = {}
    for entry in args.specialist:
        if "=" not in entry:
            print(f"ignoring specialist arg without '=': {entry}", file=sys.stderr)
            continue
        crop_name, _, checkpoint_path = entry.partition("=")
        crop_name = crop_name.strip().lower()
        path = Path(checkpoint_path.strip())
        if crop_name and path.exists():
            specialists.setdefault(crop_name, []).append(path)
    if args.apple_specialist and Path(args.apple_specialist).exists():
        specialists.setdefault("apple", []).append(Path(args.apple_specialist))

    result = cascade_predict(
        Path(args.image_path),
        Path(args.unified),
        specialists,
        threshold=args.threshold,
        routing_threshold=args.routing_threshold,
        tta_passes=args.tta_passes,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0)
