"""Training loop implementation for staged fine-tuning."""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ml.plant.dl.classifier.data import SplitBundle, create_splits, load_manifest, make_loader
from ml.plant.dl.classifier.metrics import compute_metrics, save_metrics
from ml.plant.dl.classifier.modeling import create_model
from ml.plant.dl.classifier.paths import (
    CHECKPOINT_DIR,
    OUTPUT_ROOT,
    PRODUCTION_DIR,
    PRODUCTION_METADATA_PATH,
    PRODUCTION_MODEL_PATH,
    REPORT_DIR,
)


@dataclass(slots=True)
class TrainConfig:
    model_name: str = "tf_efficientnetv2_s"
    image_size: int = 300
    batch_size: int = 16
    epochs_stage1: int = 8
    epochs_stage2: int = 5
    learning_rate_stage1: float = 3e-4
    learning_rate_stage2: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    confidence_threshold: float = 0.65
    output_name: str = "best_model"
    promote_to_production: bool = False
    grad_accumulation_steps: int = 1
    amp_enabled: bool = True
    early_stopping_patience: int = 3
    input_mode: str = "full_image"
    augmentation_profile: str = "standard"
    roi_cache_dir: str | None = None
    field_source_boost: float = 1.0
    stage2_controlled_fraction: float = 0.20
    loss_name: str = "cross_entropy"
    focal_gamma: float = 2.0
    target_column: str = "canonical_label"
    crop_filter: str | None = None
    field_val_size: float = 0.0
    select_stage2_on_field: bool = False
    extra_field_sources: tuple[str, ...] = ()


class FocalLoss(nn.Module):
    def __init__(self, *, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_term = (1.0 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "best_model"


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_one_epoch(model, loader, optimizer, criterion, device, *, grad_accumulation_steps: int, use_amp: bool, scaler):
    model.train()
    running_loss = 0.0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(tqdm(loader, leave=False), start=1):
        images = images.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        loss_value = loss.detach().item()
        loss = loss / max(grad_accumulation_steps, 1)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % max(grad_accumulation_steps, 1) == 0 or step == len(loader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss_value * images.size(0)
        total += images.size(0)
    return running_loss / max(total, 1)


def _evaluate(model, loader, criterion, device, target_names, *, use_amp: bool):
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, target_names)
    metrics["loss"] = running_loss / max(total, 1)
    return metrics


def _make_loss(frame: pd.DataFrame, class_to_index: dict[str, int], device: torch.device, config: TrainConfig):
    class_names = np.asarray(sorted(class_to_index, key=lambda label: class_to_index[label]))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_names,
        y=frame[config.target_column].tolist(),
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    if config.loss_name == "focal":
        return FocalLoss(weight=weight_tensor, gamma=config.focal_gamma)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def _fit_stage(
    *,
    model,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    class_to_index: dict[str, int],
    target_names: list[str],
    config: TrainConfig,
    lr: float,
    epochs: int,
    device: torch.device,
    stage_name: str,
):
    stage_field_boost = config.field_source_boost if stage_name == "stage2_field_finetune" else 1.0
    train_loader = make_loader(
        train_frame,
        class_to_index,
        config.image_size,
        config.batch_size,
        True,
        config.num_workers,
        input_mode=config.input_mode,
        augmentation_profile=config.augmentation_profile if stage_name == "stage2_field_finetune" else "standard",
        roi_cache_root=config.roi_cache_dir,
        field_source_boost=stage_field_boost,
        label_column=config.target_column,
    )
    val_loader = make_loader(
        val_frame,
        class_to_index,
        config.image_size,
        config.batch_size,
        False,
        config.num_workers,
        input_mode=config.input_mode,
        augmentation_profile="standard",
        roi_cache_root=config.roi_cache_dir,
        label_column=config.target_column,
    )
    criterion = _make_loss(train_frame, class_to_index, device, config)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = config.amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_accumulation_steps=config.grad_accumulation_steps,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = _evaluate(model, val_loader, criterion, device, target_names, use_amp=use_amp)
        scheduler.step()

        history.append(
            {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return model, history


def run_training(config: TrainConfig) -> dict:
    _seed_everything(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    splits: SplitBundle = create_splits(
        manifest,
        seed=config.seed,
        stage2_controlled_fraction=config.stage2_controlled_fraction,
        label_column=config.target_column,
        crop_filter=config.crop_filter,
        field_val_size=config.field_val_size,
        extra_field_sources=tuple(config.extra_field_sources),
    )
    training_manifest = manifest.copy()
    if config.crop_filter:
        training_manifest = training_manifest[training_manifest["crop"] == config.crop_filter].copy()
    classes = sorted(training_manifest[config.target_column].dropna().unique().tolist())
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    target_names = classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config.model_name, num_classes=len(classes), pretrained=True).to(device)
    output_name = _slugify_name(config.output_name)

    model, stage1_history = _fit_stage(
        model=model,
        train_frame=splits.stage1_train,
        val_frame=splits.stage1_val,
        class_to_index=class_to_index,
        target_names=target_names,
        config=config,
        lr=config.learning_rate_stage1,
        epochs=config.epochs_stage1,
        device=device,
        stage_name="stage1_controlled",
    )

    # By default stage 2 is still selected on the controlled-val split. When
    # select_stage2_on_field is set and a field-val split exists, pick the
    # stage-2 checkpoint by field-val macro-F1 instead, so the model is chosen
    # for field generalization rather than clean-lab accuracy.
    stage2_val_frame = splits.stage1_val
    if config.select_stage2_on_field and splits.stage2_val is not None and not splits.stage2_val.empty:
        stage2_val_frame = splits.stage2_val

    model, stage2_history = _fit_stage(
        model=model,
        train_frame=splits.stage2_train,
        val_frame=stage2_val_frame,
        class_to_index=class_to_index,
        target_names=target_names,
        config=config,
        lr=config.learning_rate_stage2,
        epochs=config.epochs_stage2,
        device=device,
        stage_name="stage2_field_finetune",
    )

    if config.roi_cache_dir:
        Path(config.roi_cache_dir).mkdir(parents=True, exist_ok=True)

    criterion = _make_loss(splits.stage2_train, class_to_index, device, config)
    stage1_test_loader = make_loader(
        splits.stage1_test,
        class_to_index,
        config.image_size,
        config.batch_size,
        False,
        config.num_workers,
        input_mode=config.input_mode,
        augmentation_profile="standard",
        roi_cache_root=config.roi_cache_dir,
        label_column=config.target_column,
    )
    stage2_test_loader = make_loader(
        splits.stage2_test,
        class_to_index,
        config.image_size,
        config.batch_size,
        False,
        config.num_workers,
        input_mode=config.input_mode,
        augmentation_profile="standard",
        roi_cache_root=config.roi_cache_dir,
        label_column=config.target_column,
    )
    use_amp = config.amp_enabled and device.type == "cuda"
    controlled_metrics = _evaluate(model, stage1_test_loader, criterion, device, target_names, use_amp=use_amp)
    field_metrics = _evaluate(model, stage2_test_loader, criterion, device, target_names, use_amp=use_amp)

    checkpoint_path = CHECKPOINT_DIR / f"{output_name}.pt"
    payload = {
        "model_name": config.model_name,
        "image_size": config.image_size,
        "class_to_index": class_to_index,
        "config": asdict(config),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)

    production_model_path: str | None = None
    if config.promote_to_production:
        shutil.copy2(checkpoint_path, PRODUCTION_MODEL_PATH)
        production_model_path = str(PRODUCTION_MODEL_PATH)

    summary = {
        "output_name": output_name,
        "checkpoint_path": str(checkpoint_path),
        "production_model_path": production_model_path,
        "device": str(device),
        "controlled_test": controlled_metrics,
        "field_test": field_metrics,
        "history": stage1_history + stage2_history,
    }
    summary_path = REPORT_DIR / f"{output_name}_training_summary.json"
    controlled_metrics_path = REPORT_DIR / f"{output_name}_controlled_test_metrics.json"
    field_metrics_path = REPORT_DIR / f"{output_name}_field_test_metrics.json"
    summary["summary_path"] = str(summary_path)
    summary["controlled_metrics_path"] = str(controlled_metrics_path)
    summary["field_metrics_path"] = str(field_metrics_path)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_metrics(controlled_metrics, controlled_metrics_path)
    save_metrics(field_metrics, field_metrics_path)

    # Maintain the legacy default filenames for the main single-run workflow.
    if output_name == "best_model":
        (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_metrics(controlled_metrics, REPORT_DIR / "controlled_test_metrics.json")
        save_metrics(field_metrics, REPORT_DIR / "field_test_metrics.json")

    if config.promote_to_production:
        PRODUCTION_METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
