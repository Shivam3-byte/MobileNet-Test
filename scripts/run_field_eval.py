from src.classification.evaluate import evaluate_checkpoint
from src.utils.metrics import pretty_print_metrics


if __name__ == "__main__":
    metrics = evaluate_checkpoint(
        config_path="configs/mobilenetv3_classification.yaml",
        checkpoint_path="models/checkpoints/mobilenetv3/plantvillage_baseline/best.pt",
        data_dir="data/processed/field_cls/all",
        output_json="results/metrics/field_eval_before_finetune.json",
    )
    print(pretty_print_metrics(metrics))

