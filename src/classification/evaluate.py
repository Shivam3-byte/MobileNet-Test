from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.classification.dataset import build_imagefolder_dataset
from src.classification.model import build_model, resolve_device
from src.utils.metrics import compute_classification_metrics


def evaluate_checkpoint(config_path: str, checkpoint_path: str, data_dir: str, output_json: str | None = None):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset = build_imagefolder_dataset(data_dir, train=False, image_size=config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    device = resolve_device(config["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=len(dataset.classes), model_name=config["model_name"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    metrics = compute_classification_metrics(y_true, y_pred, dataset.classes)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(metrics.__dict__, handle, indent=2)
    return metrics

