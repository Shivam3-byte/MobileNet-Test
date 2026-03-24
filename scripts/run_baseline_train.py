from src.classification.train import train_from_config


if __name__ == "__main__":
    checkpoint = train_from_config("configs/mobilenetv3_classification.yaml", run_name="plantvillage_baseline")
    print(f"Saved best checkpoint to {checkpoint}")

