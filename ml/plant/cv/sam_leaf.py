"""Optional SAM-based leaf isolation for the CV pipeline.

Colour-based leaf masking cannot separate a green leaf from a green background
(field foliage, unripe fruit). SAM segments by object shape, not colour, so it
isolates the leaf even on green-on-green scenes. This module is a thin, *optional*
wrapper: if SAM, its weights, or torch are unavailable — or if disabled via
CROPCARE_USE_SAM=0 — every function returns None and the caller falls back to the
classical ``build_leaf_mask``. It must never raise into the pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts" / "sam" / "sam_vit_b_01ec64.pth"


def _enabled() -> bool:
    return os.environ.get("CROPCARE_USE_SAM", "1").strip().lower() not in ("0", "false", "no", "off")


@lru_cache(maxsize=1)
def _get_predictor():
    """Load SAM once per process. Returns None if anything is missing."""
    if not _enabled() or not _CHECKPOINT.exists():
        return None
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint=str(_CHECKPOINT))
        sam.to("cuda" if torch.cuda.is_available() else "cpu")
        return SamPredictor(sam)
    except Exception:
        return None


def _pick_mask(masks, scores, *, lo: float = 0.05, hi: float = 0.97):
    """Choose the most leaf-like mask: good SAM score, sensible area fraction."""
    best, best_key = None, -1.0
    for m, s in zip(masks, scores):
        area = float(m.mean())
        if area < lo or area > hi:
            continue
        key = float(s) + 0.3 * min(area, 0.6)  # prefer confident, reasonably large
        if key > best_key:
            best_key, best = key, m
    return best


def sam_leaf_mask(image_array: np.ndarray) -> np.ndarray | None:
    """Return a boolean leaf mask via SAM, or None to signal 'fall back to classical'.

    The leaf is assumed roughly centred (the pipeline localises it first), so we
    prompt with the centre point and, if that mask is too small, a central box.
    """
    predictor = _get_predictor()
    if predictor is None:
        return None
    try:
        img = np.asarray(image_array, dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        h, w = img.shape[:2]
        predictor.set_image(img)

        point = np.array([[w / 2.0, h / 2.0]])
        masks, scores, _ = predictor.predict(
            point_coords=point, point_labels=np.array([1]), multimask_output=True
        )
        best = _pick_mask(masks, scores)

        if best is None or float(best.mean()) < 0.08:
            box = np.array([w * 0.08, h * 0.08, w * 0.92, h * 0.92], dtype=np.float32)
            bmasks, bscores, _ = predictor.predict(box=box[None, :], multimask_output=False)
            cand = _pick_mask(bmasks, bscores)
            if cand is not None:
                best = cand

        if best is None:
            return None
        return best.astype(bool)
    except Exception:
        return None
