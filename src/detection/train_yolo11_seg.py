from __future__ import annotations

from ultralytics import YOLO


def train_yolo11n_seg(data_config: str, epochs: int = 50, imgsz: int = 640):
    model = YOLO("yolo11n-seg.pt")
    return model.train(data=data_config, epochs=epochs, imgsz=imgsz)

