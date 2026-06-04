"""Run inference on a single image with the trained CropCare AI model."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.classifier.label_schema import CANONICAL_LABELS
from ml.plant.dl.classifier.modeling import create_model
from ml.plant.dl.classifier.open_set_gate import evaluate_crop_support, load_open_set_gate
from ml.plant.dl.classifier.paths import PRODUCTION_MODEL_PATH, PRODUCTION_OPEN_SET_GATE_PATH


def load_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_index = checkpoint["class_to_index"]
    index_to_class = {index: label for label, index in class_to_index.items()}
    model = create_model(checkpoint["model_name"], num_classes=len(class_to_index), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["image_size"], index_to_class, checkpoint["config"]


def _make_tta_views(image: Image.Image, tta_passes: int) -> list[Image.Image]:
    views = [image]
    if tta_passes >= 2:
        views.append(TF.hflip(image))
    if tta_passes >= 3:
        views.append(TF.vflip(image))
    if tta_passes >= 4:
        views.append(TF.hflip(TF.vflip(image)))
    return views[: max(tta_passes, 1)]


def _aggregate_crop_probabilities(
    probs: torch.Tensor,
    index_to_class: dict[int, str],
) -> dict[str, float]:
    crop_probabilities: dict[str, float] = {}
    for class_index, label in index_to_class.items():
        info = CANONICAL_LABELS[label]
        crop_probabilities[info.crop] = crop_probabilities.get(info.crop, 0.0) + float(probs[int(class_index)])
    return crop_probabilities


def predict(image_path: Path, checkpoint_path: Path, threshold: float, tta_passes: int = 1):
    model, image_size, index_to_class, config = load_checkpoint(checkpoint_path)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tta_views = _make_tta_views(image, tta_passes)
    tensor = torch.stack([transform(view) for view in tta_views], dim=0)
    with torch.no_grad():
        features = model.forward_features(tensor)
        pre_logits = model.forward_head(features, pre_logits=True)
        logits = model.forward_head(features)
        probs = torch.softmax(logits, dim=1).mean(dim=0)
        embedding = F.normalize(pre_logits, dim=1).mean(dim=0)
        embedding = F.normalize(embedding, dim=0)
    confidence, index = torch.max(probs, dim=0)
    label = index_to_class[int(index)]
    info = CANONICAL_LABELS[label]
    top_indices = torch.topk(probs, k=min(3, probs.shape[0])).indices.tolist()
    top_predictions = [
        {
            "label": index_to_class[int(class_index)],
            "probability": float(probs[int(class_index)]),
        }
        for class_index in top_indices
    ]
    class_probabilities = {
        index_to_class[int(class_index)]: float(probs[int(class_index)])
        for class_index in range(probs.shape[0])
    }
    crop_probabilities = _aggregate_crop_probabilities(probs, index_to_class)
    crop_support = evaluate_crop_support(embedding, load_open_set_gate(PRODUCTION_OPEN_SET_GATE_PATH))
    is_known_crop = True if crop_support is None else bool(crop_support["is_known_crop"])

    predicted_label = label
    predicted_crop = info.crop
    predicted_disease = info.disease
    is_healthy = info.is_healthy
    health_status = "healthy" if info.is_healthy else ("uncertain" if float(confidence) < threshold else "diseased")

    if not is_known_crop:
        predicted_label = "unknown_leaf_crop"
        predicted_crop = "unknown_leaf_crop"
        predicted_disease = None
        is_healthy = False
        health_status = "uncertain"

    result = {
        "image_path": str(image_path),
        "predicted_label": predicted_label,
        "crop": predicted_crop,
        "disease": predicted_disease,
        "is_healthy": is_healthy,
        "confidence": float(confidence),
        "health_status": health_status,
        "threshold": threshold,
        "model_name": config["model_name"],
        "tta_passes": len(tta_views),
        "top_predictions": top_predictions,
        "class_probabilities": class_probabilities,
        "crop_probabilities": crop_probabilities,
        "is_known_crop": is_known_crop,
        "matched_supported_label": label,
        "matched_supported_crop": info.crop,
        "open_set_gate": crop_support,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-image inference.")
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default=str(PRODUCTION_MODEL_PATH))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--tta-passes", type=int, default=1)
    args = parser.parse_args()

    output = predict(Path(args.image_path), Path(args.checkpoint), args.threshold, tta_passes=args.tta_passes)
    print(json.dumps(output, indent=2))
