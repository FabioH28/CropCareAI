"""Segmentation evaluation for the CropCare AI lesion-segmentation methods.

Two modes:

1. AGREEMENT MODE (default, fully automatic).
   PlantDoc has no ground-truth lesion masks, so this does NOT measure accuracy.
   It quantifies how much the six segmentation methods AGREE with each other and
   with the 2-of-3 consensus (pairwise Dice/IoU + per-method coverage). This is a
   robustness / consensus-design-justification analysis, and is labelled as such.

2. GROUND-TRUTH MODE (--gt-dir DIR).
   If you hand-label a few lesion masks (binary PNG, white = lesion), drop them in
   DIR named `<image-stem>.png`. This then computes REAL Dice/IoU/precision/recall
   of each method (and the consensus) against your masks. Even 10-15 labelled
   leaves give a defensible accuracy number.

Outputs a JSON report and a comparison figure under ml/plant/artifacts/reports/.

Usage:
    python scripts/eval_segmentation.py                  # agreement analysis
    python scripts/eval_segmentation.py --per-crop 4
    python scripts/eval_segmentation.py --gt-dir path/to/hand_masks
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.classifier.data import load_manifest
from ml.plant.cv.segmentation import (
    build_leaf_mask,
    threshold_segmentation,
    region_growing_segmentation,
    split_and_merge_segmentation,
    clustering_segmentation,
    simple_superpixels,
    graph_cut_segmentation,
)

SUPPORTED_CROPS = ["apple", "tomato", "grape", "corn", "pepper", "potato"]
# Same three the deployed pipeline uses for the 2-of-3 consensus.
CORE_METHODS = ["threshold", "clustering", "graph_cut"]
REPORT_DIR = PROJECT_ROOT / "ml" / "plant" / "artifacts" / "reports"


def all_method_masks(rgb: np.ndarray, leaf: np.ndarray) -> dict[str, np.ndarray]:
    _, superpixel = simple_superpixels(rgb, leaf)
    return {
        "threshold": threshold_segmentation(rgb, leaf),
        "region_grow": region_growing_segmentation(rgb, leaf),
        "split_merge": split_and_merge_segmentation(rgb, leaf),
        "clustering": clustering_segmentation(rgb, leaf),
        "superpixels": superpixel,
        "graph_cut": graph_cut_segmentation(rgb, leaf),
    }


def consensus_mask(masks: dict[str, np.ndarray], leaf: np.ndarray) -> np.ndarray:
    votes = np.stack([masks[m].astype(np.uint8) for m in CORE_METHODS], axis=0).sum(axis=0)
    return (votes >= 2) & leaf


def dice_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    sa, sb = float(a.sum()), float(b.sum())
    if union == 0:  # both empty -> perfect agreement on "no lesion"
        return 1.0, 1.0
    iou = inter / union
    dice = (2 * inter) / (sa + sb) if (sa + sb) > 0 else 1.0
    return dice, iou


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def sample_images(per_crop: int) -> list[tuple[str, str]]:
    m = load_manifest()
    field = m[(m["source"] == "plantdoc") & (m["source_split"] == "test")].dropna(subset=["path", "crop"])
    out = []
    for crop in SUPPORTED_CROPS:
        rows = field[field["crop"] == crop].head(per_crop)
        out.extend((r["path"], crop) for _, r in rows.iterrows())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Segmentation agreement / ground-truth evaluation.")
    ap.add_argument("--per-crop", type=int, default=4, help="images per crop to sample (agreement mode)")
    ap.add_argument("--gt-dir", default=None, help="dir of hand-labeled binary masks <stem>.png for real IoU")
    ap.add_argument("--figure", default=str(REPORT_DIR / "figures" / "segmentation_comparison.png"))
    ap.add_argument("--output", default=str(REPORT_DIR / "segmentation_eval.json"))
    args = ap.parse_args()

    method_names = ["threshold", "region_grow", "split_merge", "clustering", "superpixels", "graph_cut"]
    samples = sample_images(args.per_crop)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    pair_dice = {f"{a}|{b}": [] for a, b in combinations(method_names, 2)}
    vs_consensus = {m: {"dice": [], "iou": []} for m in method_names}
    coverage = {m: [] for m in method_names}
    gt_scores = {m: {"dice": [], "iou": [], "precision": [], "recall": []} for m in method_names + ["consensus"]}
    gt_count = 0
    figure_panels = []

    for path, crop in samples:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        leaf = build_leaf_mask(rgb)
        if leaf.sum() == 0:
            continue
        masks = all_method_masks(rgb, leaf)
        cons = consensus_mask(masks, leaf)
        leaf_area = float(leaf.sum())

        for m in method_names:
            coverage[m].append(float(masks[m].sum()) / leaf_area * 100.0)
            d, i = dice_iou(masks[m], cons)
            vs_consensus[m]["dice"].append(d)
            vs_consensus[m]["iou"].append(i)
        for a, b in combinations(method_names, 2):
            d, _ = dice_iou(masks[a], masks[b])
            pair_dice[f"{a}|{b}"].append(d)

        # ground-truth scoring if a mask exists for this image
        if gt_dir is not None:
            gt_path = gt_dir / f"{Path(path).stem}.png"
            if gt_path.exists():
                gt = np.asarray(Image.open(gt_path).convert("L")) > 127
                gt = gt & leaf  # only score inside the leaf
                gt_count += 1
                scored = dict(masks)
                scored["consensus"] = cons
                for m, mask in scored.items():
                    d, i = dice_iou(mask, gt)
                    p, r = precision_recall(mask, gt)
                    gt_scores[m]["dice"].append(d)
                    gt_scores[m]["iou"].append(i)
                    gt_scores[m]["precision"].append(p)
                    gt_scores[m]["recall"].append(r)

        if len(figure_panels) < 3:
            figure_panels.append((rgb, leaf, masks, cons, crop))

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    report = {
        "mode": "ground_truth" if gt_dir else "agreement_only",
        "note": (
            "Agreement metrics measure inter-method CONSISTENCY, not accuracy. "
            "PlantDoc has no ground-truth lesion masks; pass --gt-dir with hand-labeled "
            "masks for true IoU/Dice."
        ),
        "n_images": len(samples),
        "core_consensus_methods": CORE_METHODS,
        "mean_coverage_percent_of_leaf": {m: mean(v) for m, v in coverage.items()},
        "mean_dice_vs_consensus": {m: mean(v["dice"]) for m, v in vs_consensus.items()},
        "mean_iou_vs_consensus": {m: mean(v["iou"]) for m, v in vs_consensus.items()},
        "mean_pairwise_dice": {k: mean(v) for k, v in pair_dice.items()},
    }
    if gt_dir:
        report["ground_truth_images_scored"] = gt_count
        report["ground_truth"] = {
            m: {k: mean(v) for k, v in d.items()} for m, d in gt_scores.items()
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def overlay(rgb, mask, color=(255, 60, 60), alpha=0.45):
            out = rgb.astype(np.float32).copy()
            m = mask.astype(bool)
            for c in range(3):
                out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
            return out.astype(np.uint8)

        rows = len(figure_panels)
        cols = 1 + len(method_names) + 1
        fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.3 * rows))
        if rows == 1:
            axes = axes[None, :]
        for ri, (rgb, leaf, masks, cons, crop) in enumerate(figure_panels):
            axes[ri, 0].imshow(rgb); axes[ri, 0].set_ylabel(crop, fontsize=9)
            axes[ri, 0].set_title("input" if ri == 0 else "", fontsize=9)
            for ci, m in enumerate(method_names, start=1):
                axes[ri, ci].imshow(overlay(rgb, masks[m]))
                if ri == 0:
                    axes[ri, ci].set_title(m, fontsize=8)
            axes[ri, cols - 1].imshow(overlay(rgb, cons, color=(255, 230, 0)))
            if ri == 0:
                axes[ri, cols - 1].set_title("consensus", fontsize=8)
            for ci in range(cols):
                axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
        plt.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=120, bbox_inches="tight")
        report["figure"] = args.figure
    except Exception as exc:  # figure is optional
        print(f"(figure skipped: {exc})", file=sys.stderr)

    # ---- console summary ----
    print(f"\n=== segmentation evaluation ({report['mode']}, {len(samples)} images) ===\n")
    print(f"{'method':12} {'coverage%':>10} {'dice vs consensus':>18}")
    for m in method_names:
        print(f"{m:12} {report['mean_coverage_percent_of_leaf'][m]:>10} {report['mean_dice_vs_consensus'][m]:>18}")
    if gt_dir:
        print(f"\n--- GROUND-TRUTH ({gt_count} labelled images) ---")
        print(f"{'method':12} {'dice':>8} {'iou':>8} {'prec':>8} {'recall':>8}")
        for m in method_names + ["consensus"]:
            g = report["ground_truth"][m]
            print(f"{m:12} {g['dice']!s:>8} {g['iou']!s:>8} {g['precision']!s:>8} {g['recall']!s:>8}")
    print(f"\nwrote {args.output}")
    if report.get("figure"):
        print(f"wrote {report['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
