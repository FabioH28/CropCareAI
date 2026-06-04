"""Shared image preprocessing helpers for smart farm CV pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage, signal


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int], method: str = "bilinear") -> Image.Image:
    methods = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return image.resize(size, methods.get(method, Image.Resampling.BILINEAR))


def to_grayscale(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image)


def to_numpy(image: Image.Image, dtype=np.float32) -> np.ndarray:
    return np.asarray(image, dtype=dtype)


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def contrast_stretch(array: np.ndarray, lower_percentile: float = 2.0, upper_percentile: float = 98.0) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    low = np.percentile(array, lower_percentile, axis=(0, 1), keepdims=True)
    high = np.percentile(array, upper_percentile, axis=(0, 1), keepdims=True)
    stretched = (array - low) / np.maximum(high - low, 1e-6)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def gamma_correction(array: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    normalized = normalize_rgb(array)
    corrected = np.power(normalized, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def histogram_equalize(gray_array: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray_array, dtype=np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256)
    cdf = histogram.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    scaled = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min() + 1e-6)
    lut = np.ma.filled(scaled, 0).astype(np.uint8)
    return lut[gray]


def gaussian_blur(array: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        return np.stack([ndimage.gaussian_filter(array[..., channel], sigma=sigma) for channel in range(array.shape[-1])], axis=-1)
    return ndimage.gaussian_filter(array, sigma=sigma)


def median_denoise(array: np.ndarray, size: int = 3) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        filtered = [ndimage.median_filter(array[..., channel], size=size) for channel in range(array.shape[-1])]
        return np.stack(filtered, axis=-1).astype(np.uint8)
    return ndimage.median_filter(array, size=size).astype(np.uint8)


def wiener_restore(array: np.ndarray, size: int = 5) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        restored = [signal.wiener(array[..., channel], (size, size)) for channel in range(array.shape[-1])]
        return np.clip(np.stack(restored, axis=-1), 0, 255).astype(np.uint8)
    restored = signal.wiener(array, (size, size))
    return np.clip(restored, 0, 255).astype(np.uint8)


def unsharp_mask(array: np.ndarray, sigma: float = 1.0, amount: float = 1.25) -> np.ndarray:
    base = np.asarray(array, dtype=np.float32)
    blurred = gaussian_blur(base, sigma=sigma)
    sharpened = base + amount * (base - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def rgb_to_hsv_array(array: np.ndarray) -> np.ndarray:
    rgb = normalize_rgb(array)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)
    valid = delta > 1e-6

    red_mask = valid & (maximum == red)
    green_mask = valid & (maximum == green)
    blue_mask = valid & (maximum == blue)

    hue[red_mask] = ((green[red_mask] - blue[red_mask]) / delta[red_mask]) % 6.0
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2.0
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4.0
    hue = hue / 6.0

    saturation = np.zeros_like(maximum)
    non_zero = maximum > 1e-6
    saturation[non_zero] = delta[non_zero] / maximum[non_zero]

    value = maximum
    return np.stack([hue, saturation, value], axis=-1).astype(np.float32)
