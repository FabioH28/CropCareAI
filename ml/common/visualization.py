"""Visualization helpers used by ML scripts and CV analysis pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ml.common.utils import min_max_normalize


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.ndim == 3 and array.shape[-1] == 3 and array.max() <= 1.0:
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)
    scaled = min_max_normalize(array)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_array_as_image(array: np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(normalize_to_uint8(array))
    image.save(path)
    return path


def heatmap(array: np.ndarray) -> np.ndarray:
    values = min_max_normalize(array)
    red = np.clip(1.8 * values - 0.2, 0, 1)
    green = np.clip(1.8 * (1 - np.abs(values - 0.5) * 2.0), 0, 1)
    blue = np.clip(1.3 - 1.8 * values, 0, 1)
    return np.stack([red, green, blue], axis=-1)


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    canvas = np.zeros(mask.shape + (3,), dtype=np.uint8)
    canvas[mask] = np.asarray(color, dtype=np.uint8)
    return canvas


def overlay_mask(
    image_array: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 80, 60),
    alpha: float = 0.42,
) -> np.ndarray:
    base = normalize_to_uint8(image_array).astype(np.float32)
    mask_rgb = colorize_mask(mask, color).astype(np.float32)
    selector = np.asarray(mask, dtype=bool)[..., None]
    blended = np.where(selector, (1 - alpha) * base + alpha * mask_rgb, base)
    return np.clip(blended, 0, 255).astype(np.uint8)


def overlay_boundaries(
    image_array: np.ndarray,
    labels: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    image = normalize_to_uint8(image_array).copy()
    labels = np.asarray(labels)
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundaries[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    image[boundaries] = np.asarray(color, dtype=np.uint8)
    return image


def stack_images_grid(
    labeled_images: list[tuple[str, np.ndarray]],
    columns: int = 2,
    cell_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if not labeled_images:
        raise ValueError("At least one image is required.")

    prepared: list[np.ndarray] = []
    width, height = cell_size or (None, None)
    for _, array in labeled_images:
        image = Image.fromarray(normalize_to_uint8(array))
        if width and height:
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        prepared.append(np.asarray(image))

    rows = int(np.ceil(len(prepared) / columns))
    image_height, image_width = prepared[0].shape[:2]
    canvas = np.zeros((rows * image_height, columns * image_width, 3), dtype=np.uint8)

    for index, image in enumerate(prepared):
        row = index // columns
        column = index % columns
        top = row * image_height
        left = column * image_width
        canvas[top : top + image_height, left : left + image_width] = image[..., :3]

    return canvas
