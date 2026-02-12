# Fine-Tuning: MedGemma 4B Entity Resolution Classifier

Fine-tuned MedGemma 4B (text-only) for pairwise patient entity resolution on Synthea synthetic medical records. Uses QLoRA (r=16, attention + MLP targets) with a sequence classification head.

**Performance:** F1 = 0.963, Accuracy = 0.963, Precision = 0.969, Recall = 0.958 (6,482-pair test set)

## Model

- HF Hub: [`abicyclerider/medgemma-4b-entity-resolution-text-only`](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only)
- Architecture: `Gemma3TextForSequenceClassification` (3.88B params, LoRA merged, no vision tower)
- Dataset: [`abicyclerider/entity-resolution-pairs`](https://huggingface.co/datasets/abicyclerider/entity-resolution-pairs) (30K train / 6.5K eval / 6.5K test, balanced)

## Pipeline

```
prepare_dataset.py → train_classifier_on_gpu.py → export_text_only_model.py
```

1. **`prepare_dataset.py`** — Loads Synthea data via `shared/`, generates Strategy D summaries, builds balanced train/eval/test splits, pushes to HF Hub
2. **`train_classifier_on_gpu.py`** — QLoRA fine-tuning on GPU (H100 ~2.2h, L40S needs gradient checkpointing)
3. **`export_text_only_model.py`** — Merges LoRA adapter, strips vision tower, uploads merged model

## Inference

```bash
# Evaluate on HF test split
python inference_classifier.py --dataset

# Classify a custom CSV (must have a 'text' column)
python inference_classifier.py --input-csv pairs.csv --output-csv predictions.csv
```

Requires CUDA GPU. Uses 4-bit NF4 quantization by default (`--no-quantize` for bf16).

## Setup

```bash
cd fine-tuning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training on RunPod, see `RUNPOD_GUIDE.md`.
