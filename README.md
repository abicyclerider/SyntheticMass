# SyntheticMass

Entity resolution pipeline for synthetic medical records, built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge).

## Overview

SyntheticMass generates realistic multi-facility patient data from [Synthea](https://github.com/synthetichealth/synthea), injects demographic errors and duplicates, then resolves matching records back into a master patient index. The system combines probabilistic record linkage with a fine-tuned MedGemma clinical language model to achieve near-perfect entity resolution on synthetic data.

The pipeline is fully reproducible via [DVC](https://dvc.org/), with separate tracks for model training and inference. GPU stages run on [RunPod](https://www.runpod.io/) via automated scripts.

## Architecture

Entity resolution uses a three-tier approach: fast probabilistic linkage handles the clear cases, while a clinical language model resolves the ambiguous middle.

**Tier 1 — Splink probabilistic linkage.** [Splink v4](https://github.com/moj-analytical-services/splink) trains a Fellegi-Sunter model via unsupervised EM on 7 demographic fields (names, address, city, ZIP, SSN, date of birth). Blocking rules generate candidate pairs; each pair gets a match probability. Pairs above 0.95 are auto-matched, below 0.05 are auto-rejected.

**Tier 2 — MedGemma gray zone classifier.** Pairs in the 0.05–0.95 range are ambiguous on demographics alone. For each, the pipeline generates structured clinical summaries (conditions, medications, allergies, observations, procedures grouped by year) and passes them to a fine-tuned [MedGemma 4B](https://huggingface.co/google/medgemma-4b-it) text-only classifier. The model was QLoRA-trained (r=16, attention + MLP targets) on 30K balanced pairs, reducing 4.2B parameters to a 3.88B text-only model by stripping the vision tower before training.

**Tier 3 — Logit-space fusion.** The Splink match probability and LLM prediction logit are combined in log-odds space (`w_splink * splink_logit + w_llm * llm_logit`), with a configurable Splink probability floor that vetoes LLM matches when demographics strongly disagree. Final matches are clustered via connected components and merged into golden records with field-level conflict resolution.

## Results

### Pipeline Performance (500-patient inference set)

| Stage | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Splink auto-match (>0.95) | 0.993 | 0.998 | 0.996 |
| Full pipeline (Splink + MedGemma + fusion) | **0.993** | **1.000** | **0.997** |

- **1,184** true pairs recovered, **8** false positives, **0** false negatives
- **599** golden records produced vs **600** true patients
- **100%** blocking recall — no true pairs lost at the candidate generation stage
- **406** gray zone pairs classified by MedGemma, **2** additional matches recovered

### Classifier Performance (held-out test set)

| Model | Params | F1 | Precision | Recall |
|-------|--------|----|-----------|--------|
| MedGemma 4B text-only (QLoRA, merged) | 3.88B | 0.963 | 0.969 | 0.958 |

**HuggingFace Hub:**
- Merged model: [`abicyclerider/medgemma-4b-entity-resolution-text-only`](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only)
- Dataset: [`abicyclerider/entity-resolution-pairs`](https://huggingface.co/datasets/abicyclerider/entity-resolution-pairs) (30K train / 6.5K eval / 6.5K test)

## Pipeline

The pipeline is managed by DVC and defined in [`dvc.yaml`](dvc.yaml). Parameters are in [`params.yaml`](params.yaml).

**Inference track** (main pipeline):

```
generate → augment → resolve → infer → golden_records
```

1. **generate** — Run Synthea to create synthetic patient CSVs (500 patients, seed 67890)
2. **augment** — Distribute patients across 5 facilities, inject demographic errors, create ground truth
3. **resolve** — Splink probabilistic linkage: auto-matches + gray zone pairs with clinical summaries
4. **infer** — MedGemma classifier scores gray zone pairs (RunPod GPU)
5. **golden_records** — Fuse auto-matches + LLM predictions, cluster, merge into golden records, evaluate

**Training track** (model fine-tuning):

```
generate_training → augment_training → prepare_dataset → train → export
```

6. **generate_training** — Separate Synthea run (500 patients, seed 12345)
7. **augment_training** — Augment training data with same error model
8. **prepare_dataset** — Build HuggingFace dataset with Strategy D clinical summaries
9. **train** — QLoRA fine-tuning on RunPod GPU (H100 ~2h, 3 epochs)
10. **export** — Merge LoRA adapter into base model, push to HF Hub

## Project Structure

```
SyntheticMass/
├── synthea_runner/          # Synthea patient data generation (Docker submodule)
├── augmentation/            # Error injection, duplicates, ground truth labels
├── entity_resolution/       # Splink blocking, pair generation, golden records
│   └── core/                # Core algorithms (splink_linker, golden_record, evaluation)
├── llm_classifier/          # MedGemma 4B classifier: data prep, training, export, inference
├── shared/                  # Shared utilities (data_loader, summarizer, ground_truth)
├── output/                  # DVC-managed pipeline outputs (gitignored)
├── dvc.yaml                 # Pipeline definition (10 stages)
├── params.yaml              # Pipeline parameters
└── pyproject.toml           # Python project config (ruff, mypy, pytest)
```

## Getting Started

### Prerequisites

- **Docker** — for Synthea data generation and entity resolution
- **Python 3.11+** — for augmentation, entity resolution, and dataset preparation
- **GPU (48GB+ VRAM)** — for fine-tuning and inference (H100 or L40S recommended)
- **DVC** — for pipeline orchestration (`pip install dvc`)

### Quick Start

```bash
# Install Python dependencies
pip install -e .

# Initialize Synthea submodule
git submodule update --init --recursive

# Run the full inference pipeline
dvc repro golden_records

# Run a single stage
dvc repro resolve

# View the pipeline DAG
dvc dag
```

GPU stages (`infer`, `train`, `export`) run on [RunPod](https://www.runpod.io/) via automated scripts. See [`llm_classifier/README.md`](llm_classifier/README.md) for setup details.

## License

See `LICENSE` file for details.
