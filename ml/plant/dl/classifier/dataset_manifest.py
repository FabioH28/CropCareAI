"""Dataset manifest builder for CropCare AI training."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from ml.plant.dl.classifier.label_schema import normalize_label
from ml.plant.dl.classifier.paths import (
    IMAGE_SUFFIXES,
    MANIFEST_PATH,
    MENDELEY_ROOT,
    OUTPUT_ROOT,
    PLANTDOC_ROOT,
    PLANTVILLAGE_ROOT,
    PP2020_ROOT,
    PP2021_ROOT,
)


@dataclass(slots=True)
class ManifestRow:
    path: str
    source: str
    source_split: str
    raw_label: str
    canonical_label: str
    crop: str
    disease: str
    is_healthy: bool
    domain: str


def _iter_images(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _plantvillage_rows() -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    if not PLANTVILLAGE_ROOT.exists():
        return rows

    for class_dir in sorted(PLANTVILLAGE_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        info = normalize_label("plantvillage", class_dir.name)
        if info is None:
            continue
        for image_path in _iter_images(class_dir):
            rows.append(
                ManifestRow(
                    path=str(image_path),
                    source="plantvillage",
                    source_split="raw",
                    raw_label=class_dir.name,
                    canonical_label=info.canonical_label,
                    crop=info.crop,
                    disease=info.disease,
                    is_healthy=info.is_healthy,
                    domain="controlled",
                )
            )
    return rows


def _plantdoc_rows() -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    if not PLANTDOC_ROOT.exists():
        return rows

    for split in ("train", "test"):
        split_root = PLANTDOC_ROOT / split
        if not split_root.exists():
            continue
        for class_dir in sorted(split_root.iterdir()):
            if not class_dir.is_dir():
                continue
            info = normalize_label("plantdoc", class_dir.name)
            if info is None:
                continue
            for image_path in _iter_images(class_dir):
                rows.append(
                    ManifestRow(
                        path=str(image_path),
                        source="plantdoc",
                        source_split=split,
                        raw_label=class_dir.name,
                        canonical_label=info.canonical_label,
                        crop=info.crop,
                        disease=info.disease,
                        is_healthy=info.is_healthy,
                        domain="field",
                    )
                )
    return rows


def _mendeley_rows() -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    if not MENDELEY_ROOT.exists():
        return rows

    extracted_candidates = [path for path in MENDELEY_ROOT.rglob("*") if path.is_dir() and any(child.is_dir() for child in path.iterdir())]
    for root in extracted_candidates:
        if root.name.endswith(".7z"):
            continue
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            info = normalize_label("mendeley_tomato", class_dir.name)
            if info is None:
                continue
            for image_path in _iter_images(class_dir):
                rows.append(
                    ManifestRow(
                        path=str(image_path),
                        source="mendeley_tomato",
                        source_split="raw",
                        raw_label=class_dir.name,
                        canonical_label=info.canonical_label,
                        crop=info.crop,
                        disease=info.disease,
                        is_healthy=info.is_healthy,
                        domain="controlled",
                    )
                )
    return rows


def _plant_pathology_2021_rows() -> list[ManifestRow]:
    """Plant Pathology 2021 (Kaggle FGVC8): real-world apple field photos.

    Expects layout:
        plant_pathology_2021/train.csv          (image,labels columns)
        plant_pathology_2021/train_images/<id>.jpg

    Only single-label rows whose label is one of {healthy, scab, rust} are
    kept. Multi-label rows ("scab frog_eye_leaf_spot", "complex", etc.) and
    classes outside our canonical schema are skipped.
    """
    rows: list[ManifestRow] = []
    if not PP2021_ROOT.exists():
        return rows

    csv_path = PP2021_ROOT / "train.csv"
    images_root = PP2021_ROOT / "train_images"
    if not csv_path.exists() or not images_root.exists():
        return rows

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_label = (row.get("labels") or "").strip()
            image_id = (row.get("image") or "").strip()
            if not raw_label or not image_id:
                continue
            # multi-label rows have spaces between labels
            if " " in raw_label:
                continue
            info = normalize_label("plant_pathology_2021", raw_label)
            if info is None:
                continue
            image_path = images_root / image_id
            if not image_path.exists():
                continue
            rows.append(
                ManifestRow(
                    path=str(image_path),
                    source="plant_pathology_2021",
                    source_split="train",
                    raw_label=raw_label,
                    canonical_label=info.canonical_label,
                    crop=info.crop,
                    disease=info.disease,
                    is_healthy=info.is_healthy,
                    domain="field",
                )
            )
    return rows


def _plant_pathology_2020_rows() -> list[ManifestRow]:
    """Plant Pathology 2020 (Kaggle FGVC7): real-world apple field photos.

    Expects layout:
        plant_pathology_2020/train.csv          (one-hot columns)
        plant_pathology_2020/images/Train_<id>.jpg

    train.csv columns: image_id,healthy,multiple_diseases,scab,rust (0/1).
    Only single-class rows (exactly one of healthy/scab/rust set to 1) are
    kept. multiple_diseases rows are skipped.
    """
    rows: list[ManifestRow] = []
    if not PP2020_ROOT.exists():
        return rows

    csv_path = PP2020_ROOT / "train.csv"
    images_root = PP2020_ROOT / "images"
    if not csv_path.exists() or not images_root.exists():
        return rows

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            single_label: str | None = None
            multi = False
            for column in ("healthy", "scab", "rust"):
                value = (row.get(column) or "0").strip()
                if value == "1":
                    if single_label is not None:
                        multi = True
                        break
                    single_label = column
            if multi or single_label is None:
                continue
            if (row.get("multiple_diseases") or "0").strip() == "1":
                continue
            info = normalize_label("plant_pathology_2020", single_label)
            if info is None:
                continue
            image_path = images_root / f"{image_id}.jpg"
            if not image_path.exists():
                continue
            rows.append(
                ManifestRow(
                    path=str(image_path),
                    source="plant_pathology_2020",
                    source_split="train",
                    raw_label=single_label,
                    canonical_label=info.canonical_label,
                    crop=info.crop,
                    disease=info.disease,
                    is_healthy=info.is_healthy,
                    domain="field",
                )
            )
    return rows


def build_manifest() -> list[ManifestRow]:
    rows = (
        _plantvillage_rows()
        + _plantdoc_rows()
        + _mendeley_rows()
        + _plant_pathology_2021_rows()
        + _plant_pathology_2020_rows()
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(ManifestRow.__annotations__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return rows


if __name__ == "__main__":
    rows = build_manifest()
    print(f"Wrote {len(rows)} rows to {MANIFEST_PATH}")
