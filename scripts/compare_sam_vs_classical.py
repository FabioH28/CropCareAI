"""Modern-vs-classical leaf-segmentation comparison.

Compares the project's hand-built classical leaf segmentation
(``ml.plant.cv.segmentation.build_leaf_mask``) against a modern
foundation model (Segment Anything / SAM, ViT-B) on the same images.

Design / fairness notes
-----------------------
- Leaf segmentation is the one task where a *zero-shot* foundation model has a
  fair shot: SAM segments generic objects, it has no concept of "disease
  lesion", so we compare on the leaf-object mask, not the lesion mask.
- SAM is run as an *interactive* model: it is prompted with the tight bounding
  box that the classical pipeline's own leaf localizer produces. This is a
  deliberate synergy — the hand-built CV proposes the region, the foundation
  model refines the pixel mask — and it means the comparison answers the honest
  question: "given the same ROI, how closely does my from-scratch leaf mask
  agree with a state-of-the-art learned segmenter?"
- Agreement (IoU / Dice) is therefore a *validation* of the classical method,
  not a contest the classical method is expected to "win".

Outputs (under ml/plant/artifacts/reports/sam_vs_classical/):
- one side-by-side PNG per image (original | classical | SAM | agreement)
- a montage grid PNG
- sam_vs_classical_summary.json with per-image and mean IoU / Dice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.cv.segmentation import build_leaf_mask  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "ml/plant/artifacts/sam/sam_vit_b_01ec64.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml/plant/artifacts/reports/sam_vs_classical"
PLANTDOC_TEST = PROJECT_ROOT / "datasets/plant/plantdoc/test"

# One representative field image per supported crop.
DEFAULT_CATEGORIES = [
    "Apple Scab Leaf",
    "Tomato Early blight leaf",
    "grape leaf black rot",
    "Corn leaf blight",
    "Potato leaf early blight",
    "Bell_pepper leaf spot",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: Path, max_side: int = 1024) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = max_side / float(max(width, height))
    if scale < 1.0:
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    total = float(a.sum() + b.sum())
    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / total if total > 0 else 0.0
    return iou, dice


def agreement_rgb(image: np.ndarray, classical: np.ndarray, sam: np.ndarray) -> np.ndarray:
    """Dim the image, then paint: green=both, red=classical-only, blue=SAM-only."""
    base = (image.astype(np.float32) * 0.45).astype(np.uint8)
    both = classical & sam
    classical_only = classical & ~sam
    sam_only = sam & ~classical
    base[both] = (0, 220, 90)
    base[classical_only] = (235, 60, 60)
    base[sam_only] = (60, 120, 240)
    return base


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def pick_default_images() -> list[Path]:
    picked: list[Path] = []
    for category in DEFAULT_CATEGORIES:
        folder = PLANTDOC_TEST / category
        if not folder.is_dir():
            continue
        for candidate in sorted(folder.iterdir()):
            if candidate.suffix.lower() in IMAGE_SUFFIXES:
                picked.append(candidate)
                break
    return picked


def build_sam_predictor(checkpoint: Path, device: str):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam.to(device)
    return SamPredictor(sam)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="Image paths. If omitted, one PlantDoc image per crop is used.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images] if args.images else pick_default_images()
    if not image_paths:
        print("No images found to process.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(
            f"SAM checkpoint not found at: {checkpoint}\n"
            "Download it (375 MB) with:\n"
            "  curl -L -o ml/plant/artifacts/sam/sam_vit_b_01ec64.pth \\\n"
            "    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            "and ensure the package is installed: pip install segment-anything",
            file=sys.stderr,
        )
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  images: {len(image_paths)}")
    predictor = build_sam_predictor(checkpoint, device)

    records: list[dict] = []
    panels: list[tuple[str, np.ndarray]] = []

    for index, path in enumerate(image_paths, start=1):
        image = load_image(path, max_side=args.max_side)
        classical_mask = build_leaf_mask(image).astype(bool)

        bbox = mask_bbox(classical_mask)
        if bbox is None:
            print(f"[{index}/{len(image_paths)}] {path.name}: classical found no leaf, skipping.")
            continue

        predictor.set_image(image)
        box = np.array(bbox, dtype=np.float32)[None, :]
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        sam_mask = masks[0].astype(bool)

        iou, dice = iou_dice(classical_mask, sam_mask)
        record = {
            "image": str(path),
            "image_name": path.name,
            "category": path.parent.name,
            "classical_leaf_pixels": int(classical_mask.sum()),
            "sam_leaf_pixels": int(sam_mask.sum()),
            "sam_score": float(scores[0]),
            "iou": round(iou, 4),
            "dice": round(dice, 4),
        }
        records.append(record)
        print(f"[{index}/{len(image_paths)}] {path.name}: IoU={iou:.3f} Dice={dice:.3f}")

        # Per-image side-by-side figure.
        agree = agreement_rgb(image, classical_mask, sam_mask)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
        axes[0].imshow(image)
        axes[0].set_title("Original (field photo)")
        axes[1].imshow(overlay(image, classical_mask, (235, 60, 60)))
        axes[1].set_title("Classical (hand-built)")
        axes[2].imshow(overlay(image, sam_mask, (60, 120, 240)))
        axes[2].set_title(f"SAM ViT-B (score {scores[0]:.2f})")
        axes[3].imshow(agree)
        axes[3].set_title(f"Agreement  IoU={iou:.2f}  Dice={dice:.2f}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{record['category']}  —  {path.name}", fontsize=11)
        fig.tight_layout()
        out_png = output_dir / f"{index:02d}_{path.stem[:40]}_sam_vs_classical.png"
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)

        panels.append((record["category"], agree))

    if not records:
        print("Nothing processed.", file=sys.stderr)
        return 1

    mean_iou = round(float(np.mean([r["iou"] for r in records])), 4)
    mean_dice = round(float(np.mean([r["dice"] for r in records])), 4)

    # Montage of the agreement panels.
    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, panel), rec in zip(axes, panels, records):
        ax.imshow(panel)
        ax.set_title(f"{title}\nIoU={rec['iou']:.2f} Dice={rec['dice']:.2f}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        f"SAM (learned) vs hand-built classical leaf segmentation\n"
        f"green=agree  red=classical-only  blue=SAM-only   |   mean IoU={mean_iou}  mean Dice={mean_dice}",
        fontsize=12,
    )
    fig.tight_layout()
    montage_png = output_dir / "00_montage_sam_vs_classical.png"
    fig.savefig(montage_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "method_classical": "hand-built consensus leaf mask (ml.plant.cv.segmentation.build_leaf_mask)",
        "method_learned": "Segment Anything (SAM) ViT-B, box-prompted with the classical leaf bbox",
        "comparison_task": "leaf-object segmentation (zero-shot-fair task)",
        "num_images": len(records),
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "per_image": records,
        "interpretation": (
            "High agreement indicates the from-scratch classical leaf segmentation tracks a "
            "state-of-the-art foundation model closely. SAM is prompted with the classical "
            "pipeline's own leaf bounding box, so the score reflects mask-level agreement within a "
            "shared ROI, not an unconstrained head-to-head."
        ),
    }
    summary_path = output_dir / "sam_vs_classical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nMean IoU={mean_iou}  Mean Dice={mean_dice}  over {len(records)} images")
    print(f"Figures + summary written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
