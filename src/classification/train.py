from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.dataset import build_imagefolder_dataset
from src.classification.model import build_model, resolve_device
from src.utils.seed import seed_everything


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def train_from_config(
    config_path: str,
    train_dir: str | None = None,
    val_dir: str | None = None,
    run_name: str = "baseline",
    init_checkpoint: str | None = None,
):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed_everything(config["seed"])
    train_root = train_dir or config["train_dir"]
    val_root = val_dir or config["val_dir"]

    train_dataset = build_imagefolder_dataset(train_root, train=True, image_size=config["image_size"])
    val_dataset = build_imagefolder_dataset(val_root, train=False, image_size=config["image_size"])

    device = resolve_device(config["device"])
    model = build_model(num_classes=len(train_dataset.classes), model_name=config["model_name"], pretrained=True).to(device)
    init_checkpoint = init_checkpoint or config.get("init_checkpoint")
    if init_checkpoint and Path(init_checkpoint).exists():
        checkpoint = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    best_acc = 0.0
    checkpoint_dir = Path(config["checkpoint_dir"]) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_dataset.classes,
                    "config": config,
                },
                checkpoint_dir / "best.pt",
            )

    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return str(checkpoint_dir / "best.pt")
