"""Dataset and dataloader helpers."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from ml.plant.cv.segmentation import build_leaf_mask, extract_leaf_roi
from ml.plant.dl.classifier.paths import MANIFEST_PATH, ROI_CACHE_ROOT

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


@dataclass(slots=True)
class SplitBundle:
    stage1_train: pd.DataFrame
    stage1_val: pd.DataFrame
    stage1_test: pd.DataFrame
    stage2_train: pd.DataFrame
    stage2_test: pd.DataFrame
    stage2_val: pd.DataFrame | None = None


def _slugify(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"_", "-", "."} else "_" for character in value)


def _cache_key(path_value: str, input_mode: str) -> str:
    payload = f"{input_mode}::{Path(path_value).resolve()}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


class CropDiseaseDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_index: dict[str, int],
        transform=None,
        *,
        label_column: str = "canonical_label",
        input_mode: str = "full_image",
        roi_cache_root: str | Path | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform
        self.label_column = label_column
        self.input_mode = input_mode
        self.roi_cache_root = Path(roi_cache_root) if roi_cache_root else None
        if self.roi_cache_root is not None:
            self.roi_cache_root.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.frame)

    def _prepare_variant(self, image: Image.Image, image_path: str) -> Image.Image:
        if self.input_mode == "full_image":
            return image.convert("RGB")

        if self.roi_cache_root is not None:
            cache_path = self.roi_cache_root / f"{_cache_key(image_path, self.input_mode)}.png"
            if cache_path.exists():
                try:
                    return Image.open(cache_path).convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError):
                    cache_path.unlink(missing_ok=True)

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        leaf_mask = build_leaf_mask(rgb)
        roi = extract_leaf_roi(rgb, leaf_mask, padding_ratio=0.12)
        if self.input_mode == "leaf_context":
            prepared = roi["context_crop"] if roi["leaf_detected"] else rgb
        elif self.input_mode == "leaf_isolated":
            prepared = roi["isolated_crop"] if roi["leaf_detected"] else rgb
        else:
            prepared = rgb

        prepared_image = Image.fromarray(prepared.astype(np.uint8)).convert("RGB")
        if self.roi_cache_root is not None:
            temp_cache_path = cache_path.with_name(
                f"{cache_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png"
            )
            prepared_image.save(temp_cache_path)
            temp_cache_path.replace(cache_path)
        return prepared_image

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self._prepare_variant(image, str(row["path"]))
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_index[row[self.label_column]]
        return image, label


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run dl/prepare_dataset.py first.")
    return pd.read_csv(MANIFEST_PATH)


def _clean_labeled_frame(
    frame: pd.DataFrame,
    class_to_index: dict[str, int] | None = None,
    *,
    label_column: str = "canonical_label",
) -> pd.DataFrame:
    cleaned = frame.dropna(subset=[label_column, "path"]).copy()
    if class_to_index is not None:
        cleaned = cleaned[cleaned[label_column].isin(class_to_index)].copy()
    return cleaned.reset_index(drop=True)


def _sample_stage2_controlled(
    stage1_train: pd.DataFrame,
    seed: int,
    controlled_fraction: float,
    *,
    label_column: str = "canonical_label",
) -> pd.DataFrame:
    sampled_groups: list[pd.DataFrame] = []
    for _, group in stage1_train.groupby(label_column):
        fraction = min(max(controlled_fraction, 0.01), 1.0)
        sample_size = len(group) if len(group) <= 5 else max(1, int(np.ceil(len(group) * fraction)))
        sampled_groups.append(group.sample(n=sample_size, random_state=seed))

    if not sampled_groups:
        return stage1_train.iloc[0:0].copy().reset_index(drop=True)

    return pd.concat(sampled_groups, ignore_index=True)


def create_splits(
    manifest: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.10,
    seed: int = 42,
    stage2_controlled_fraction: float = 0.20,
    *,
    label_column: str = "canonical_label",
    crop_filter: str | None = None,
    field_val_size: float = 0.0,
    extra_field_sources: tuple[str, ...] = (),
) -> SplitBundle:
    if label_column not in manifest.columns:
        raise ValueError(f"Manifest does not contain target label column: {label_column}")

    if crop_filter:
        if "crop" not in manifest.columns:
            raise ValueError("Manifest does not contain a crop column for crop_filter.")
        manifest = manifest[manifest["crop"] == crop_filter].copy()

    manifest = _clean_labeled_frame(manifest, label_column=label_column)
    if manifest.empty:
        raise ValueError("No usable samples remain after applying label/crop filters.")

    controlled = manifest[manifest["source"].isin(["plantvillage", "mendeley_tomato"])].copy()
    field_train = manifest[(manifest["source"] == "plantdoc") & (manifest["source_split"] == "train")].copy()
    field_test = manifest[(manifest["source"] == "plantdoc") & (manifest["source_split"] == "test")].copy()
    if controlled.empty:
        raise ValueError("No controlled samples are available for stage 1 training.")
    if field_test.empty:
        raise ValueError("No field test samples are available for evaluation.")

    stage1_train, stage1_test = train_test_split(
        controlled,
        test_size=test_size,
        random_state=seed,
        stratify=controlled[label_column],
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    stage1_train, stage1_val = train_test_split(
        stage1_train,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=stage1_train[label_column],
    )

    # Optionally hold out a field-only validation split (never overlapping the
    # field test set) so stage-2 checkpoint selection can target field
    # generalization instead of clean-lab accuracy. This is the fix for
    # specialists that score high on PlantVillage but collapse on PlantDoc.
    stage2_val = None
    if field_val_size and not field_train.empty:
        class_counts = field_train[label_column].value_counts()
        can_stratify = len(class_counts) > 1 and bool((class_counts >= 2).all())
        try:
            field_train, stage2_val = train_test_split(
                field_train,
                test_size=field_val_size,
                random_state=seed,
                stratify=field_train[label_column] if can_stratify else None,
            )
        except ValueError:
            stage2_val = None

    # Pull in extra real-field sources (e.g. Plant Pathology 2020/2021 apple
    # photos) as additional stage-2 training data. These are added AFTER the
    # field-val carve so validation/selection stays on the PlantDoc
    # distribution that matches the field test, while training still benefits
    # from the larger, more varied real-field set.
    if extra_field_sources:
        extra = manifest[manifest["source"].isin(extra_field_sources)].copy()
        if not extra.empty:
            field_train = pd.concat([field_train, extra], ignore_index=True)

    mixed_controlled = _sample_stage2_controlled(
        stage1_train,
        seed,
        controlled_fraction=stage2_controlled_fraction,
        label_column=label_column,
    )
    stage2_train = _clean_labeled_frame(
        pd.concat([field_train, mixed_controlled], ignore_index=True),
        label_column=label_column,
    )

    return SplitBundle(
        stage1_train=_clean_labeled_frame(stage1_train, label_column=label_column),
        stage1_val=_clean_labeled_frame(stage1_val, label_column=label_column),
        stage1_test=_clean_labeled_frame(stage1_test, label_column=label_column),
        stage2_train=stage2_train,
        stage2_test=_clean_labeled_frame(field_test, label_column=label_column),
        stage2_val=(
            _clean_labeled_frame(stage2_val, label_column=label_column)
            if stage2_val is not None
            else None
        ),
    )


def build_transforms(image_size: int, augmentation_profile: str = "standard"):
    if augmentation_profile == "field_heavy":
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size + 48, image_size + 48)),
                transforms.RandomResizedCrop(image_size, scale=(0.60, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(28),
                transforms.RandomPerspective(distortion_scale=0.16, p=0.22),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.88, 1.10), shear=8),
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.28, hue=0.04),
                transforms.RandomAutocontrast(p=0.18),
                transforms.RandomAdjustSharpness(sharpness_factor=1.7, p=0.18),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.22),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.18, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def make_loader(
    frame: pd.DataFrame,
    class_to_index: dict[str, int],
    image_size: int,
    batch_size: int,
    training: bool,
    num_workers: int,
    *,
    input_mode: str = "full_image",
    augmentation_profile: str = "standard",
    roi_cache_root: str | Path | None = None,
    field_source_boost: float = 1.0,
    label_column: str = "canonical_label",
) -> DataLoader:
    frame = _clean_labeled_frame(frame, class_to_index, label_column=label_column)
    if frame.empty:
        raise ValueError("Cannot build a dataloader from an empty frame after label sanitization.")

    pin_memory = torch.cuda.is_available()
    train_transform, eval_transform = build_transforms(image_size, augmentation_profile=augmentation_profile)
    cache_root = None
    if roi_cache_root is not None and input_mode != "full_image":
        cache_root = Path(roi_cache_root)
    elif input_mode != "full_image":
        cache_root = ROI_CACHE_ROOT / _slugify(input_mode)

    dataset = CropDiseaseDataset(
        frame,
        class_to_index,
        transform=train_transform if training else eval_transform,
        label_column=label_column,
        input_mode=input_mode,
        roi_cache_root=cache_root,
    )

    if training:
        class_counts = frame[label_column].value_counts()
        sample_weights = frame[label_column].map(lambda label: 1.0 / class_counts[label]).to_numpy(dtype=np.float32)
        if field_source_boost > 1.0 and "source" in frame.columns:
            source_weights = np.where(frame["source"].to_numpy() == "plantdoc", field_source_boost, 1.0).astype(np.float32)
            sample_weights = sample_weights * source_weights
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
