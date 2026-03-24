from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from src.classification.model import build_model, resolve_device


def load_classifier(checkpoint_path: str, device_name: str = "auto"):
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names), model_name=config["model_name"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = Compose(
        [
            Resize((config["image_size"], config["image_size"])),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, class_names, transform, device


def predict_image(checkpoint_path: str, image_path: str):
    model, class_names, transform, device = load_classifier(checkpoint_path)
    image = Image.open(Path(image_path)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    class_idx = int(probs.argmax().item())
    return {
        "predicted_class": class_names[class_idx],
        "confidence": float(probs[class_idx].item()),
        "all_scores": {class_names[i]: float(score) for i, score in enumerate(probs.cpu().tolist())},
    }

