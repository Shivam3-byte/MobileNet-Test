from src.classification.train import train_from_config


if __name__ == "__main__":
    checkpoint = train_from_config(
        "configs/mobilenetv3_classification.yaml",
        train_dir="data/processed/field_cls/train",
        val_dir="data/processed/field_cls/val",
        run_name="field_finetune",
        init_checkpoint="models/checkpoints/mobilenetv3/plantvillage_baseline/best.pt",
    )
    print(f"Saved fine-tuned checkpoint to {checkpoint}")
