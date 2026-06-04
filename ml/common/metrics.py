"""Generic metric and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.common.utils import json_ready, safe_divide


def write_metrics(metrics: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    return path


def mask_coverage(mask: np.ndarray, support_mask: np.ndarray | None = None) -> float:
    mask = np.asarray(mask, dtype=bool)
    if support_mask is None:
        support_mask = np.ones_like(mask, dtype=bool)
    else:
        support_mask = np.asarray(support_mask, dtype=bool)
    covered = np.logical_and(mask, support_mask).sum()
    total = support_mask.sum()
    return safe_divide(float(covered), float(total), default=0.0)


def jaccard_index(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return safe_divide(float(intersection), float(union), default=1.0)


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    return safe_divide(float(2 * intersection), float(total), default=1.0)


def agreement_ratio(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 1.0
    scores: list[float] = []
    for index, mask_a in enumerate(masks):
        for mask_b in masks[index + 1 :]:
            scores.append(jaccard_index(mask_a, mask_b))
    return float(np.mean(scores)) if scores else 1.0


def severity_from_ratio(infected_ratio: float) -> str:
    if infected_ratio < 0.03:
        return "low"
    if infected_ratio < 0.12:
        return "medium"
    if infected_ratio < 0.28:
        return "high"
    return "critical"


def urgency_from_status(health_status: str, severity_level: str) -> str:
    if health_status == "healthy":
        return "low"
    if severity_level == "critical":
        return "urgent"
    if severity_level == "high":
        return "high"
    if health_status == "diseased":
        return "medium"
    if health_status == "suspicious" and severity_level == "medium":
        return "medium"
    return "low"
