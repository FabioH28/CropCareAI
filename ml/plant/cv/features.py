"""Handcrafted feature extraction helpers for plant images."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from ml.common.metrics import agreement_ratio, mask_coverage
from ml.common.preprocessing import normalize_rgb
from ml.common.transforms import compute_brownness_index, compute_excess_green
from ml.common.utils import safe_divide


def color_statistics(image: Image.Image | np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        array = np.asarray(image, dtype=np.float32)

    if mask is None:
        pixels = array.reshape(-1, array.shape[-1])
    else:
        pixels = array[np.asarray(mask, dtype=bool)]

    if pixels.size == 0:
        return {
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "std_r": 0.0,
            "std_g": 0.0,
            "std_b": 0.0,
        }

    channel_means = pixels.mean(axis=0)
    channel_stds = pixels.std(axis=0)
    return {
        "mean_r": float(channel_means[0]),
        "mean_g": float(channel_means[1]),
        "mean_b": float(channel_means[2]),
        "std_r": float(channel_stds[0]),
        "std_g": float(channel_stds[1]),
        "std_b": float(channel_stds[2]),
    }


def texture_features(gray_array: np.ndarray, mask: np.ndarray | None = None, levels: int = 16) -> dict[str, float]:
    gray = np.asarray(gray_array, dtype=np.float32)
    if mask is None:
        mask = np.ones_like(gray, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    quantized = np.clip((gray / 256.0) * levels, 0, levels - 1).astype(np.int32)
    matrix = np.zeros((levels, levels), dtype=np.float64)

    valid = mask[:, :-1] & mask[:, 1:]
    left = quantized[:, :-1][valid]
    right = quantized[:, 1:][valid]
    for value_left, value_right in zip(left, right, strict=False):
        matrix[value_left, value_right] += 1
        matrix[value_right, value_left] += 1

    total = matrix.sum()
    if total <= 0:
        return {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0, "entropy": 0.0}

    probabilities = matrix / total
    rows, cols = np.indices(probabilities.shape)
    contrast = np.sum(((rows - cols) ** 2) * probabilities)
    homogeneity = np.sum(probabilities / (1.0 + np.abs(rows - cols)))
    energy = np.sum(probabilities**2)
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    return {
        "contrast": float(contrast),
        "homogeneity": float(homogeneity),
        "energy": float(energy),
        "entropy": float(entropy),
    }


def shape_features(mask: np.ndarray, support_mask: np.ndarray | None = None) -> dict[str, float]:
    binary = np.asarray(mask, dtype=bool)
    support = np.ones_like(binary, dtype=bool) if support_mask is None else np.asarray(support_mask, dtype=bool)
    area = float(np.logical_and(binary, support).sum())
    total = float(support.sum())
    eroded = ndimage.binary_erosion(binary)
    perimeter = float(np.logical_and(binary, ~eroded).sum())
    labeled, count = ndimage.label(binary)
    component_areas = ndimage.sum(binary, labeled, range(1, count + 1)) if count else np.array([])
    dominant_area = float(component_areas.max()) if component_areas.size else 0.0
    compactness = safe_divide(perimeter * perimeter, 4.0 * np.pi * max(area, 1.0))
    return {
        "lesion_area_ratio": safe_divide(area, total),
        "perimeter_pixels": perimeter,
        "component_count": float(count),
        "dominant_component_ratio": safe_divide(dominant_area, area),
        "compactness": compactness,
    }


def extract_feature_bundle(
    image_array: np.ndarray,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    gray_array: np.ndarray,
    frequency_data: dict[str, np.ndarray],
    wavelet_data: dict[str, np.ndarray],
    segmentation_masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    rgb = normalize_rgb(image_array)
    exg = compute_excess_green(rgb)
    brownness = compute_brownness_index(rgb)
    sobel_edges = np.hypot(ndimage.sobel(gray_array, axis=0), ndimage.sobel(gray_array, axis=1))

    low_energy = float(np.mean(np.abs(frequency_data["low_pass"])))
    high_energy = float(np.mean(np.abs(frequency_data["high_pass"])))
    wavelet_energy = {name: float(np.mean(np.square(values))) for name, values in wavelet_data.items()}

    return {
        "leaf_color": color_statistics(rgb * 255.0, leaf_mask),
        "lesion_color": color_statistics(rgb * 255.0, lesion_mask),
        "texture": texture_features(gray_array, leaf_mask),
        "lesion_texture": texture_features(gray_array, lesion_mask),
        "shape": shape_features(lesion_mask, leaf_mask),
        "indices": {
            "mean_excess_green": float(exg[leaf_mask].mean()) if np.any(leaf_mask) else 0.0,
            "mean_brownness": float(brownness[leaf_mask].mean()) if np.any(leaf_mask) else 0.0,
            "lesion_brownness": float(brownness[lesion_mask].mean()) if np.any(lesion_mask) else 0.0,
            "edge_density": float(sobel_edges[leaf_mask].mean()) if np.any(leaf_mask) else 0.0,
            "infected_area_percentage": mask_coverage(lesion_mask, leaf_mask) * 100.0,
        },
        "frequency": {
            "low_frequency_energy": low_energy,
            "high_frequency_energy": high_energy,
            "high_to_low_ratio": safe_divide(high_energy, low_energy),
        },
        "wavelets": wavelet_energy,
        "segmentation": {
            "method_agreement": agreement_ratio(list(segmentation_masks.values())),
            "method_coverage_mean": float(
                np.mean([mask_coverage(mask, leaf_mask) for mask in segmentation_masks.values()])
            ),
        },
    }
