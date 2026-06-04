"""Shared transform builders and image-transform helpers."""

from __future__ import annotations

import numpy as np

from ml.common.preprocessing import normalize_rgb
from ml.common.utils import min_max_normalize


def build_train_transform(image_size: int):
    from torchvision import transforms as tv_transforms

    return tv_transforms.Compose(
        [
            tv_transforms.Resize((image_size + 32, image_size + 32)),
            tv_transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.RandomVerticalFlip(),
            tv_transforms.RandomRotation(20),
            tv_transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
            tv_transforms.RandomApply([tv_transforms.GaussianBlur(kernel_size=3)], p=0.15),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size: int):
    from torchvision import transforms as tv_transforms

    return tv_transforms.Compose(
        [
            tv_transforms.Resize((image_size, image_size)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def compute_excess_green(image_array: np.ndarray) -> np.ndarray:
    rgb = normalize_rgb(image_array)
    return 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]


def compute_brownness_index(image_array: np.ndarray) -> np.ndarray:
    rgb = normalize_rgb(image_array)
    numerator = rgb[..., 0] + rgb[..., 2] - 2.0 * rgb[..., 1]
    denominator = np.maximum(rgb.sum(axis=-1), 1e-6)
    return numerator / denominator


def fft_magnitude_spectrum(gray_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = np.asarray(gray_array, dtype=np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(fft))
    return fft, min_max_normalize(magnitude)


def frequency_filter(gray_array: np.ndarray, radius_ratio: float = 0.12) -> dict[str, np.ndarray]:
    gray = np.asarray(gray_array, dtype=np.float32)
    fft_shifted = np.fft.fftshift(np.fft.fft2(gray))
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    radius = max(4, int(min(rows, cols) * radius_ratio))
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

    low_mask = distance <= radius
    high_mask = ~low_mask

    low_pass = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shifted * low_mask)))
    high_pass = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shifted * high_mask)))

    return {
        "fft_shifted": fft_shifted,
        "low_pass": low_pass,
        "high_pass": high_pass,
        "low_mask": low_mask.astype(np.uint8),
        "high_mask": high_mask.astype(np.uint8),
    }


def haar_wavelet_decompose(gray_array: np.ndarray) -> dict[str, np.ndarray]:
    gray = np.asarray(gray_array, dtype=np.float32)
    if gray.shape[0] % 2:
        gray = gray[:-1, :]
    if gray.shape[1] % 2:
        gray = gray[:, :-1]

    rows_low = (gray[:, 0::2] + gray[:, 1::2]) / 2.0
    rows_high = (gray[:, 0::2] - gray[:, 1::2]) / 2.0

    ll = (rows_low[0::2, :] + rows_low[1::2, :]) / 2.0
    lh = (rows_low[0::2, :] - rows_low[1::2, :]) / 2.0
    hl = (rows_high[0::2, :] + rows_high[1::2, :]) / 2.0
    hh = (rows_high[0::2, :] - rows_high[1::2, :]) / 2.0

    return {
        "approximation": ll,
        "horizontal": lh,
        "vertical": hl,
        "diagonal": hh,
    }
