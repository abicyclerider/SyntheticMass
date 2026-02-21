# ClinFuse: Patient Entity Resolution Powered by MedGemma Clinical Reasoning

**Kaggle MedGemma Impact Challenge** | [Paper](paper/clinfuse.pdf) | [Model](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only) | [Dataset](https://huggingface.co/datasets/abicyclerider/entity-resolution-pairs) | [Code](https://github.com/abicyclerider/clinfuse)

ClinFuse is an entity resolution pipeline that uses a fine-tuned MedGemma 4B model to match patient records across healthcare facilities. It combines fast probabilistic linkage (Splink) with MedGemma's clinical reasoning to resolve ambiguous cases that demographics alone cannot, producing a deduplicated Master Patient Index. See the [paper](paper/clinfuse.pdf) for the full problem statement, technical approach, and deployment analysis.

## Results

Evaluated on 2,922 synthetic patients (6,264 records across 5 facilities) under adversarial conditions: 8--12 demographic errors per record and clinical histories split across facilities.

| Method | Precision | Recall | F1 | Split Patients |
|--------|-----------|--------|----|----------------|
| Splink only (P > 0.99) | 0.910 | 0.467 | 0.617 | 993 |
| Splink best threshold (P > 0.15) | 0.787 | 0.744 | 0.764 | 421 |
| **ClinFuse (Splink + MedGemma)** | **0.922** | **0.910** | **0.916** | **115** |

MedGemma recovered 2,702 additional patient matches from 151,309 ambiguous gray-zone pairs. Split patients reduced by 88%. Cluster completeness improved from 66% to 96%.

| Model | Params | F1 | Precision | Recall |
|-------|--------|----|-----------|--------|
| MedGemma 4B text-only (QLoRA fine-tuned) | 3.88B | 0.963 | 0.969 | 0.958 |

## Resources

| Resource | Link |
|----------|------|
| Fine-tuned model | [abicyclerider/medgemma-4b-entity-resolution-text-only](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only) |
| Text-only base model | [abicyclerider/medgemma-4b-text-only-base](https://huggingface.co/abicyclerider/medgemma-4b-text-only-base) |
| Training dataset | [abicyclerider/entity-resolution-pairs](https://huggingface.co/datasets/abicyclerider/entity-resolution-pairs) (90K train / 19K eval / 19K test) |
| Paper | [`paper/clinfuse.pdf`](paper/clinfuse.pdf) |
| Source code | [github.com/abicyclerider/clinfuse](https://github.com/abicyclerider/clinfuse) |

## Reproducing Results

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Docker | 20.10+ | All pipeline stages run in containers |
| DVC | 3.x | Pipeline orchestration (`pip install dvc`) |
| Python | 3.11+ | Local development, tests, and DVC driver scripts |
| RAM | 8 GB+ | Synthea generation uses `-Xmx8g` for JVM |
| Disk | ~10 GB | For generated data, Docker images, and outputs |
| GPU (remote) | 48 GB+ VRAM | H100 recommended for training (~2.2h) and inference |

### Setup

```bash
git clone https://github.com/abicyclerider/clinfuse.git
cd clinfuse
pip install -e .
git submodule update --init --recursive   # Synthea Docker image
```

### API credentials

| Key | Location | Required for | How to get |
|-----|----------|-------------|------------|
| `HF_TOKEN` | `llm_classifier/.env` | GPU stages (`infer`, `train`, `export`) and `prepare_dataset` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `RUNPOD_API_KEY` | `~/.runpod/config.toml` | GPU stages (pod provisioning) | [runpod.io/console/user/settings](https://www.runpod.io/console/user/settings) |

Non-GPU stages (`generate` through `resolve`, plus `golden_records`) require **no API keys** and run entirely locally in Docker.

Set up HuggingFace token:
```bash
echo 'HF_TOKEN=hf_yourtoken' > llm_classifier/.env
```

Set up RunPod (install `runpodctl` first):
```bash
runpodctl config --apiKey YOUR_KEY
```

### Running the inference pipeline

```bash
dvc repro golden_records
```

This runs all inference stages in order:

| # | Stage | Runs on | Output |
|---|-------|---------|--------|
| 1 | `generate` | Docker (CPU) | `synthea_runner/output/synthea_raw/csv/` |
| 2 | `csv_to_parquet` | Docker (CPU) | `synthea_runner/output/synthea_parquet/` |
| 3 | `segment` | Docker (CPU) | `output/segmented/` |
| 4 | `inject_errors` | Docker (CPU) | `output/augmented/` |
| 5 | `resolve` | Docker (CPU) | `output/resolved/` (auto_matches, gray_zone_pairs, features) |
| 6 | `infer` | RunPod GPU | `output/inferred/predictions.parquet` |
| 7 | `golden_records` | Docker (CPU) | `output/golden_records/` (MPI + evaluation_metrics.json) |

You can also run a single stage: `dvc repro resolve`.

### Two-phase GPU stages

The `infer`, `train`, and `export` stages use a **launch-then-collect** pattern because GPU work runs on RunPod pods, not locally. Each stage exits with code 1 after launching its pod --- this is expected behavior, not an error.

**Workflow:**
```bash
dvc repro infer    # Phase 1: uploads data to HF Hub, launches RunPod pod, exits 1
# Wait for the pod to finish (check: runpodctl get pod)
dvc repro infer    # Phase 2: detects results on HF Hub, downloads predictions, exits 0
```

The scripts auto-detect which phase they're in via a timestamp file in `llm_classifier/.state/`. Phase detection:
1. **No timestamp file** --- LAUNCH: upload input, launch pod, exit 1
2. **Timestamp exists, HF Hub not updated** --- NOT READY: print status, exit 1
3. **Timestamp exists, HF Hub updated** --- COLLECT: download results, validate, exit 0

After a successful collect, the timestamp file is removed so the next `dvc repro` runs from scratch.

**GHCR image prerequisite:** GPU stages pull a pre-built Docker image from GHCR. The `build_gpu_image` DVC stage verifies the image exists for the current commit. This image is built automatically by CI on push to `main`. If you've made local changes to `llm_classifier/`, push to `main` and wait for CI before running GPU stages.

### Running the training pipeline

The training track generates a larger dataset (30K patients), prepares training pairs, fine-tunes MedGemma, and exports the merged model:

```bash
# Run all training stages
dvc repro export

# Stages:
#   generate_training → csv_to_parquet_training → segment_training →
#   inject_errors_training → prepare_dataset → train → export
```

The `train` and `export` stages use the same two-phase launch-then-collect pattern as `infer`. The `prepare_dataset` stage runs locally in Docker but requires `HF_TOKEN` to push the dataset to HuggingFace Hub.

After training completes, an automatic promotion gate (`promote_model.py`) compares the new model against the MLflow run history and writes `promotion_decision.json`. The `export` stage skips LoRA merging if the model was not promoted.

### Verifying results

After `dvc repro golden_records` completes:

```bash
# View final evaluation metrics
cat output/golden_records/evaluation_metrics.json

# Run tests
pytest                                         # all tests
pytest augmentation/tests -m "not slow"        # augmentation only
pytest shared/tests -m "not slow"              # shared only
pytest entity_resolution/tests                 # entity resolution

# Lint & type check
ruff check .
mypy shared/ augmentation/ entity_resolution/core/
```

The `evaluation_metrics.json` file contains precision, recall, F1, cluster completeness, split patient count, and other metrics. These should match the results table above when run with the default `params.yaml` seeds.

### Adjusting parameters

All pipeline parameters live in [`params.yaml`](params.yaml). Key parameters:

```yaml
# Inference population
generate:
  population: 2500       # Number of synthetic patients
  seed: 67890            # Synthea RNG seed

# Error injection severity
augment:
  min_errors: 8          # Min demographic errors per record
  max_errors: 12         # Max demographic errors per record

# Splink thresholds
resolve:
  auto_match_probability: 0.99   # P >= this → auto-match
  auto_reject_probability: 0.01  # P < this → auto-reject

# Training hyperparameters
train:
  epochs: 3
  lr: 0.0001
  lora_r: 16
  max_steps: 3000
```

Gray-zone fusion weights and the Bayesian prior correction are configured in [`entity_resolution/config/matching_config.yaml`](entity_resolution/config/matching_config.yaml).

### Troubleshooting

**Synthea submodule is empty:**
```bash
git submodule update --init --recursive
```

**GPU stage exits with code 1:** This is the normal launch phase. Wait for the RunPod pod to finish, then re-run the same `dvc repro` command. Check pod status with `runpodctl get pod`.

**`build_gpu_image` fails ("GHCR image not found"):** The Docker image for GPU stages is built by CI on push to `main`. Push your changes and wait for the `docker-push-gpu` workflow to complete before running GPU stages.

**HuggingFace token errors:** Ensure `llm_classifier/.env` contains `HF_TOKEN=hf_...` with a token that has read access to gated models (MedGemma requires accepting the license on HF Hub).

**RunPod pod fails to launch:** Check GPU availability at [runpod.io/console/gpu-cloud](https://www.runpod.io/console/gpu-cloud). The launch script retries 3 times with 30s delays. You can change GPU type with `--gpu-type` in `dvc.yaml`.

**OOM during training:** Default config uses H100 80GB. For smaller GPUs, reduce `train.batch_size` or `train.max_length` in `params.yaml`. QLoRA 4-bit mode can be re-enabled by removing `--no-quantize` from the `train` stage in `dvc.yaml`.

**DVC says stages are up to date:** DVC caches outputs. To force re-run: `dvc repro --force <stage>`.

## Pipeline Stages

All 18 DVC stages across two tracks:

| Stage | Track | Description |
|-------|-------|-------------|
| `build_synthea` | shared | Build Synthea Docker image |
| `build_gpu_image` | shared | Verify GHCR image exists for current commit |
| `build_prepare_dataset` | training | Build dataset preparation Docker image |
| `generate` | inference | Generate 2,500 synthetic patients via Synthea |
| `csv_to_parquet` | inference | Convert Synthea CSVs to typed Parquet tables |
| `segment` | inference | Distribute patients across 5 facilities |
| `inject_errors` | inference | Inject 8--12 demographic errors per record |
| `resolve` | inference | Splink linkage → auto-matches + gray-zone pairs |
| `infer` | inference | MedGemma batch inference on gray-zone pairs (GPU) |
| `golden_records` | inference | Fuse predictions, build MPI, evaluate |
| `evaluate_baselines` | inference | Compute Splink-only baseline metrics |
| `generate_training` | training | Generate 30,000 patients for training data |
| `csv_to_parquet_training` | training | Convert training CSVs to Parquet |
| `segment_training` | training | Distribute training patients across facilities |
| `inject_errors_training` | training | Inject 3--5 errors per training record |
| `prepare_dataset` | training | Build balanced pairs, push to HF Hub |
| `train` | training | QLoRA fine-tune MedGemma 4B (GPU) |
| `export` | training | Merge LoRA adapter, push merged model (GPU) |

## Project Structure

```
clinfuse/
├── synthea_runner/          # Synthea patient data generation (Docker submodule)
├── augmentation/            # Multi-facility distribution + demographic error injection
├── entity_resolution/       # Splink linkage, gray-zone fusion, golden record creation
│   └── core/                # Core algorithms (splink_linker, golden_record, evaluation)
├── llm_classifier/          # MedGemma fine-tuning, export, inference + RunPod orchestration
├── shared/                  # Shared utilities (data loading, clinical summarization, evaluation)
├── paper/                   # Competition paper (LaTeX)
├── analysis/                # Results analysis notebook
├── output/                  # DVC-managed pipeline outputs (gitignored)
├── dvc.yaml                 # Pipeline definition (18 stages)
├── params.yaml              # Pipeline parameters
└── pyproject.toml           # Python config (ruff, mypy, pytest)
```

## License

MIT
