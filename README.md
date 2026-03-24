# New Approach: Plant Disease Detection Workflow

This repository contains a practical workflow for a publishable student-level project on plant disease detection:

- `MobileNetV3` for disease classification
- `YOLOv8-seg` / `YOLO11n-seg` scaffolding for lesion localization
- severity estimation from segmentation masks
- Grad-CAM explainability
- before/after fine-tuning comparison on real field data

## Project Flow

1. Train a baseline classifier on PlantVillage.
2. Test the baseline directly on real field images.
3. Fine-tune on a split of the field dataset.
4. Test again on unseen field images and compare results.
5. Add segmentation-based lesion localization and severity scoring.
6. Serve predictions through a minimal API.

## Repository Layout

```text
configs/               Training and dataset configuration files
docs/                  Paper and methodology notes
scripts/               Entry points for common experiments
src/                   Core Python modules
api/                   FastAPI inference service
```

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Prepare field-data metadata:

```powershell
python scripts/prepare_field_split.py --input data/metadata/field_labels.csv --output data/metadata/split_manifest.csv
```

Train baseline:

```powershell
python scripts/run_baseline_train.py
```

Field evaluation before fine-tuning:

```powershell
python scripts/run_field_eval.py
```

Fine-tune:

```powershell
python scripts/run_finetune.py
```

Run API:

```powershell
uvicorn api.app:app --reload
```

## Notes

- Keep large datasets and checkpoints out of Git.
- Store raw PlantVillage and field images under `data/raw/`.
- The repo is designed to work first with classification, then extend to segmentation.

