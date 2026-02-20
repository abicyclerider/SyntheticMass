# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Entity resolution pipeline for synthetic medical records (Kaggle MedGemma Impact Challenge). Generates multi-facility patient data via Synthea, injects demographic errors, then resolves matching records using probabilistic linkage (Splink) + a fine-tuned MedGemma 4B classifier.

## Common Commands

### Install & Setup
```bash
pip install -e .
git submodule update --init --recursive   # Synthea submodule
```

### Run Pipeline (DVC)
```bash
dvc repro golden_records       # full inference pipeline
dvc repro resolve              # single stage
dvc dag                        # view pipeline DAG
```

### Tests
```bash
pytest                                         # all tests
pytest augmentation/tests -m "not slow"        # augmentation only, skip slow
pytest shared/tests -m "not slow"              # shared only
pytest entity_resolution/tests                 # entity resolution (all markers)
pytest augmentation/tests/unit/test_error_injection.py  # single file
pytest -k "test_name"                          # single test by name
```

### Lint & Format
```bash
ruff check .           # lint
ruff format --check .  # format check
ruff format .          # auto-format
ruff check --fix .     # auto-fix lint issues
```

### Type Check
```bash
mypy shared/ augmentation/ entity_resolution/core/
```

### Docker Builds
```bash
docker build -t augmentation augmentation/
docker build -f entity_resolution/Dockerfile -t entity_resolution .
docker build -f llm_classifier/Dockerfile.prepare -t prepare-dataset .
```

## Architecture

### Three-Tier Entity Resolution
1. **Splink probabilistic linkage** — Fellegi-Sunter EM on 7 demographic fields. Pairs above 0.99 auto-match, below 0.01 auto-reject.
2. **MedGemma gray zone classifier** — QLoRA-finetuned MedGemma 4B (text-only, vision tower stripped) classifies ambiguous pairs using structured clinical summaries.
3. **Logit-space fusion** — Combines Splink match probability + LLM prediction logit in log-odds space with a Splink probability floor veto.

### DVC Pipeline (two tracks)

**Inference:** `generate → csv_to_parquet → segment → inject_errors → resolve → infer → golden_records`

**Training:** `generate_training → csv_to_parquet_training → segment_training → inject_errors_training → prepare_dataset → train → export`

Pipeline stages run in Docker containers. GPU stages (`infer`, `train`, `export`) run on RunPod via shell scripts in `llm_classifier/`. Parameters live in `params.yaml`.

### Module Layout

- **`augmentation/`** — CSV→Parquet conversion, facility distribution, demographic error injection. CLI entry points: `augmentation.cli.csv_to_parquet`, `augmentation.cli.segment`, `augmentation.cli.inject_errors`.
- **`entity_resolution/`** — Splink linkage and golden record creation. Run as `python -m entity_resolution.resolve` and `python -m entity_resolution.build_golden_records`. Business logic in `core/` (splink_linker, golden_record, evaluation).
- **`shared/`** — Common utilities: `data_loader.py` (Parquet I/O), `summarize.py` (Strategy D clinical summaries), `ground_truth.py`, `medical_records.py`, `evaluation.py`.
- **`llm_classifier/`** — MedGemma training/inference scripts + RunPod orchestration shell scripts. Excluded from ruff/mypy. Has its own requirements files.
- **`synthea_runner/`** — Git submodule for Synthea Docker image. Excluded from ruff/mypy.

### Data Flow
```
synthea_runner/output/synthea_raw/csv/          → raw Synthea CSVs
synthea_runner/output/synthea_parquet/           → typed Parquet tables
output/segmented/facilities/facility_NNN/       → per-facility clean records
output/augmented/facilities/facility_NNN/       → records with injected errors
output/resolved/{auto_matches,gray_zone_pairs,features}.parquet
output/inferred/predictions.parquet             → MedGemma predictions
output/golden_records/                          → final MPI + evaluation metrics
```

The `output/` directory is gitignored and managed by DVC.

## Code Conventions

- Python 3.11+. Ruff for linting/formatting (line length 88). MyPy for type checking.
- Absolute imports only: `from entity_resolution.core.splink_linker import ...`, `from shared.summarize import ...`.
- CLI entry points use Click (`@click.command()`).
- Test markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`. CI skips `slow` for augmentation and shared.
- `llm_classifier/` and `synthea_runner/` are excluded from ruff and mypy (`extend-exclude` in pyproject.toml).
- `dvc.lock` is tracked in git (not gitignored).
