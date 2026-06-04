"""Open-set crop support gate built from embedding prototypes."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.plant.dl.classifier.label_schema import CANONICAL_LABELS, normalize_label
from ml.plant.dl.classifier.paths import (
    IMAGE_SUFFIXES,
    MANIFEST_PATH,
    MENDELEY_ROOT,
    PLANTDOC_ROOT,
    PLANTVILLAGE_ROOT,
    PRODUCTION_OPEN_SET_GATE_PATH,
)

SUPPORTED_CROPS = tuple(sorted({info.crop for info in CANONICAL_LABELS.values()}))
DEFAULT_MIN_SUPPORTED_SIMILARITY = 0.58
DEFAULT_UNKNOWN_MARGIN = 0.03

UNKNOWN_CROP_GROUPS: dict[str, list[Path]] = {
    "blueberry": [
        PLANTDOC_ROOT / "train" / "Blueberry leaf",
        PLANTDOC_ROOT / "test" / "Blueberry leaf",
    ],
    "cherry": [
        PLANTDOC_ROOT / "train" / "Cherry leaf",
        PLANTDOC_ROOT / "test" / "Cherry leaf",
    ],
    "peach": [
        PLANTDOC_ROOT / "train" / "Peach leaf",
        PLANTDOC_ROOT / "test" / "Peach leaf",
    ],
    "strawberry": [
        PLANTDOC_ROOT / "train" / "Strawberry leaf",
        PLANTDOC_ROOT / "test" / "Strawberry leaf",
    ],
}


def create_inference_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def extract_embedding(model: torch.nn.Module, image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    features = model.forward_features(tensor)
    pre_logits = model.forward_head(features, pre_logits=True)
    return F.normalize(pre_logits.squeeze(0), dim=0).cpu()


def _gather_existing_manifest_samples(samples_per_label: int) -> dict[str, list[Path]]:
    crop_samples: dict[str, list[Path]] = defaultdict(list)
    per_label_counts: dict[tuple[str, str], int] = defaultdict(int)

    if not MANIFEST_PATH.exists():
        return crop_samples

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = Path(row["path"])
            if not path.exists():
                continue

            crop = row["crop"]
            label = row["canonical_label"]
            key = (crop, label)
            if per_label_counts[key] >= samples_per_label:
                continue

            crop_samples[crop].append(path)
            per_label_counts[key] += 1

    return crop_samples


def _collect_directory_samples(directory: Path, limit: int) -> list[Path]:
    if not directory.exists():
        return []

    samples: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            samples.append(path)
        if len(samples) >= limit:
            break
    return samples


def _gather_supported_samples_from_directories(samples_per_label: int) -> dict[str, list[Path]]:
    crop_samples: dict[str, list[Path]] = defaultdict(list)
    per_label_counts: dict[tuple[str, str], int] = defaultdict(int)

    dataset_roots = [
        ("plantvillage", [PLANTVILLAGE_ROOT]),
        ("plantdoc", [PLANTDOC_ROOT / "train", PLANTDOC_ROOT / "test"]),
        ("mendeley_tomato", [MENDELEY_ROOT]),
    ]

    for dataset_name, roots in dataset_roots:
        for root in roots:
            if not root.exists():
                continue

            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue

                info = normalize_label(dataset_name, entry.name)
                if info is None:
                    continue

                key = (info.crop, info.canonical_label)
                remaining = samples_per_label - per_label_counts[key]
                if remaining <= 0:
                    continue

                collected = _collect_directory_samples(entry, remaining)
                if not collected:
                    continue

                crop_samples[info.crop].extend(collected)
                per_label_counts[key] += len(collected)

    return crop_samples


def collect_supported_crop_samples(samples_per_label: int = 6) -> dict[str, list[Path]]:
    crop_samples = _gather_existing_manifest_samples(samples_per_label)
    if crop_samples:
        return crop_samples
    return _gather_supported_samples_from_directories(samples_per_label)


def collect_unknown_crop_samples(samples_per_group: int = 24) -> dict[str, list[Path]]:
    crop_samples: dict[str, list[Path]] = {}
    for group_name, directories in UNKNOWN_CROP_GROUPS.items():
        samples: list[Path] = []
        for directory in directories:
            if len(samples) >= samples_per_group:
                break
            for path in _collect_directory_samples(directory, samples_per_group - len(samples)):
                samples.append(path)
        if samples:
            crop_samples[group_name] = samples
    return crop_samples


def _mean_prototype(
    model: torch.nn.Module,
    transform: transforms.Compose,
    image_paths: list[Path],
) -> torch.Tensor:
    embeddings = [extract_embedding(model, path, transform) for path in image_paths]
    stacked = torch.stack(embeddings)
    return F.normalize(stacked.mean(dim=0), dim=0).cpu()


def build_open_set_gate(
    checkpoint_path: Path,
    output_path: Path | None = None,
    *,
    samples_per_label: int = 6,
    samples_per_unknown_group: int = 24,
    minimum_supported_similarity: float = DEFAULT_MIN_SUPPORTED_SIMILARITY,
    unknown_margin: float = DEFAULT_UNKNOWN_MARGIN,
) -> dict[str, Any]:
    from ml.plant.dl.infer import load_checkpoint

    model, image_size, _index_to_class, config = load_checkpoint(checkpoint_path)
    transform = create_inference_transform(image_size)

    supported_samples = collect_supported_crop_samples(samples_per_label=samples_per_label)
    unknown_samples = collect_unknown_crop_samples(samples_per_group=samples_per_unknown_group)

    if not supported_samples:
        raise RuntimeError("No supported crop samples were found for the open-set gate.")
    if not unknown_samples:
        raise RuntimeError("No unsupported crop samples were found for the open-set gate.")

    supported_prototypes = {
        crop: _mean_prototype(model, transform, image_paths)
        for crop, image_paths in supported_samples.items()
    }
    unknown_prototypes = {
        group_name: _mean_prototype(model, transform, image_paths)
        for group_name, image_paths in unknown_samples.items()
    }

    artifact = {
        "version": "crop-support-gate-v1",
        "model_name": config["model_name"],
        "image_size": image_size,
        "embedding_dim": int(next(iter(supported_prototypes.values())).shape[0]),
        "supported_prototypes": supported_prototypes,
        "unknown_prototypes": unknown_prototypes,
        "supported_sample_counts": {crop: len(paths) for crop, paths in supported_samples.items()},
        "unknown_sample_counts": {group: len(paths) for group, paths in unknown_samples.items()},
        "decision": {
            "minimum_supported_similarity": float(minimum_supported_similarity),
            "unknown_margin": float(unknown_margin),
        },
    }

    target_path = output_path or PRODUCTION_OPEN_SET_GATE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, target_path)
    return artifact


def load_open_set_gate(gate_path: Path | None = None) -> dict[str, Any] | None:
    target_path = gate_path or PRODUCTION_OPEN_SET_GATE_PATH
    if not target_path.exists():
        return None
    return torch.load(target_path, map_location="cpu")


def evaluate_crop_support(
    embedding: torch.Tensor,
    gate_payload: dict[str, Any] | None,
    *,
    minimum_supported_similarity: float | None = None,
    unknown_margin: float | None = None,
) -> dict[str, Any] | None:
    if gate_payload is None:
        return None

    normalized_embedding = F.normalize(embedding.detach().cpu(), dim=0)
    decision_defaults = gate_payload.get("decision", {})
    min_supported = float(
        minimum_supported_similarity
        if minimum_supported_similarity is not None
        else decision_defaults.get("minimum_supported_similarity", DEFAULT_MIN_SUPPORTED_SIMILARITY)
    )
    margin = float(
        unknown_margin if unknown_margin is not None else decision_defaults.get("unknown_margin", DEFAULT_UNKNOWN_MARGIN)
    )

    supported_prototypes = gate_payload.get("supported_prototypes", {})
    unknown_prototypes = gate_payload.get("unknown_prototypes", {})
    if not supported_prototypes:
        return None

    supported_scores = {
        crop: float(torch.dot(normalized_embedding, prototype.cpu()))
        for crop, prototype in supported_prototypes.items()
    }
    unknown_scores = {
        crop: float(torch.dot(normalized_embedding, prototype.cpu()))
        for crop, prototype in unknown_prototypes.items()
    }

    top_supported_crop, top_supported_similarity = max(supported_scores.items(), key=lambda item: item[1])
    if unknown_scores:
        top_unknown_group, top_unknown_similarity = max(unknown_scores.items(), key=lambda item: item[1])
    else:
        top_unknown_group, top_unknown_similarity = None, float("-inf")

    is_known_crop = (
        top_supported_similarity >= min_supported
        and top_supported_similarity > top_unknown_similarity + margin
    )

    if is_known_crop:
        reason = "supported_crop_prototype_is_closer_than_unknown_prototypes"
    elif top_supported_similarity < min_supported:
        reason = "supported_crop_similarity_too_low"
    else:
        reason = "unknown_crop_prototype_is_closer_than_supported_crop"

    return {
        "is_known_crop": bool(is_known_crop),
        "top_supported_crop": top_supported_crop,
        "top_supported_similarity": round(top_supported_similarity, 4),
        "top_unknown_group": top_unknown_group,
        "top_unknown_similarity": round(top_unknown_similarity, 4) if top_unknown_group else None,
        "minimum_supported_similarity": round(min_supported, 4),
        "unknown_margin": round(margin, 4),
        "decision_reason": reason,
        "supported_scores": {crop: round(score, 4) for crop, score in supported_scores.items()},
        "unknown_scores": {crop: round(score, 4) for crop, score in unknown_scores.items()},
    }
