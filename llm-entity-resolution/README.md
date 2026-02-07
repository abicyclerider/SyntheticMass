# LLM Entity Resolution

LLM-based medical record comparison for entity resolution on synthetic healthcare data. This module complements the classical demographic pipeline (`entity-resolution/`) by comparing patients' medical histories only — conditions, medications, encounters, and labs — without using any demographic information.

The key architectural insight is **sequential gating**: the classical pipeline handles clear matches and non-matches, and only gray zone pairs (demographic score 4.0-6.0) are sent to the LLM. This keeps LLM costs low while adding clinical reasoning where it matters most.

## Architecture

### Sequential Gating Flow

```
Classical Pipeline (entity-resolution/)
         │
         ├── score < 4.0  ──→  Auto-reject (non-match)
         │
         ├── score >= 6.0 ──→  Auto-match
         │
         └── score 4.0-6.0 ──→  Gray Zone
                                    │
                              Medical Record
                              Summarization
                                    │
                              DSPy / MedGemma
                              ChainOfThought
                                    │
                              Match / Non-match
                              + confidence score
```

### Tool Roles

| Tool | Role |
|------|------|
| **DSPy** | Prompt optimization framework. Defines the `MedicalRecordMatchSignature`, wraps it in `ChainOfThought`, and runs MIPROv2 to optimize instructions on gray zone training data. Supports using a separate, stronger prompt model (e.g. Claude Sonnet) for instruction generation while keeping the local task model for evaluation. |
| **Promptfoo** | Prompt regression testing. Evaluates baseline vs. optimized prompts across multiple models and curated test cases. Catches regressions before deployment. |
| **Langfuse** | LLM observability. Traces every DSPy/OpenAI call with latency, token counts, and inputs/outputs for debugging and monitoring. |

### Directory Structure

```
llm-entity-resolution/
├── config/
│   ├── llm_config.yaml              # Main configuration
│   └── promptfoo/
│       ├── promptfooconfig.yaml      # Promptfoo evaluation config
│       ├── prompts/
│       │   ├── baseline_v1.txt       # Baseline prompt
│       │   └── optimized.txt         # DSPy-optimized prompt (template)
│       ├── datasets/
│       │   └── gray_zone_tests.yaml  # Curated test cases
│       └── providers/
│           └── dspy_provider.py      # Promptfoo → DSPy bridge
├── src/
│   ├── dspy_modules.py               # DSPy signature + module
│   ├── summarize.py                  # Medical record → text summary
│   ├── optimize.py                   # MIPROv2 optimization
│   ├── classify.py                   # Hybrid classification pipeline
│   ├── langfuse_setup.py             # Tracing initialization
│   └── utils.py                      # Config loading, helpers
├── data/
│   └── dspy/                         # Optimized program output (generated)
├── docker/
│   └── langfuse/
│       └── docker-compose.yaml       # Langfuse + Postgres
├── tests/
├── output/                           # Classification results (generated)
└── requirements.txt
```

## Prerequisites

1. **Ollama with MedGemma** — See [`ollama/README.md`](../ollama/README.md) for setup. At minimum, `medgemma:1.5-4b-q4-fast` must be available.

2. **Classical pipeline output** — Run the entity resolution pipeline first to generate `entity-resolution/output/predicted_matches.csv` with similarity scores.

3. **Python 3.12+** with dependencies (see Installation).

4. **API key for prompt model** (optional) — For DSPy optimization with a stronger prompt model. Set `ANTHROPIC_API_KEY` (or the relevant provider key) in your environment. Without this, optimization falls back to the local task model.

5. **Node.js** (optional) — For Promptfoo prompt testing.

6. **Docker** (optional) — For Langfuse observability.

## Installation

```bash
cd llm-entity-resolution
pip install -r requirements.txt
```

Optional — Promptfoo for prompt regression testing:

```bash
npm install -g promptfoo
```

Optional — Langfuse for LLM observability:

```bash
cd docker/langfuse
docker compose up -d
# UI at http://localhost:3000 (default login: langfuse@langfuse.com / langfuse)
```

## Usage — Developer Workflow

### 1. Optimize: tune prompts with MIPROv2

Loads gray zone pairs, generates medical history summaries, and runs DSPy's MIPROv2 optimizer to find the best instructions for the `MedicalRecordMatcher`. If a prompt model is configured (e.g. Claude Sonnet), it generates instruction candidates while MedGemma handles task evaluation.

```bash
# With a stronger prompt model (recommended):
export ANTHROPIC_API_KEY="sk-ant-..."
cd llm-entity-resolution
python -m src.optimize --config config/llm_config.yaml

# Without (falls back to task model for everything):
cd llm-entity-resolution
python -m src.optimize --config config/llm_config.yaml
```

Output: `data/dspy/optimized_program.json`

### 2. Test: evaluate prompts with Promptfoo

Runs baseline and optimized prompts against curated gray zone test cases across multiple MedGemma model variants.

```bash
cd config/promptfoo
promptfoo eval -j 1
promptfoo view
```

`-j 1` runs sequentially since Ollama serves one request at a time. The `view` command opens a browser dashboard to compare results.

### 3. Classify: run hybrid pipeline

Loads the classical pipeline's scored pairs, applies sequential gating, and runs the LLM on gray zone pairs. Evaluates against ground truth and saves results.

```bash
cd llm-entity-resolution
python -m src.classify --config config/llm_config.yaml
```

Output:
- `output/hybrid_predictions.csv` — All pairs with match decisions and source (auto_reject / auto_match / llm)
- `output/hybrid_metrics.json` — Precision, recall, F1 against ground truth

### 4. Monitor: inspect traces in Langfuse

With Langfuse running (`docker compose up -d`), every LLM call during classification is traced automatically.

Open `http://localhost:3000` to inspect:
- Individual trace inputs/outputs
- Latency and token usage per call
- Error rates across classification runs

## Configuration

All settings are in `config/llm_config.yaml`:

```yaml
run_id: "run_20260203_071928"        # Synthea augmentation run to use
base_dir: "/Users/alex/repos/Kaggle/SyntheticMass"

model:
  name: "medgemma:1.5-4b-q4-fast"    # Ollama model for classification
  api_base: "http://localhost:11434/v1"
  api_key: "ollama"                   # Required by SDK, not validated
  temperature: 0.1                    # Low for deterministic matching
  max_tokens: 256                     # Cap response length

dspy:
  optimized_program: "data/dspy/optimized_program.json"
  auto: "medium"                      # MIPROv2 search intensity
  num_trials: 15                      # Optimization trials
  max_bootstrapped_demos: 3           # Few-shot examples to bootstrap
  minibatch_size: 25                  # Examples per optimization batch
  prompt_model:                       # Optional: stronger model for instruction generation
    provider: "anthropic/claude-sonnet-4-5-20250929"
    api_key_env: "ANTHROPIC_API_KEY"  # Reads API key from this env var

langfuse:
  enabled: true                       # Set false to disable tracing
  base_url: "http://localhost:3000"
  public_key: "pk-lf-local"
  secret_key: "sk-lf-local"

gray_zone:
  lower_threshold: 4.0                # Below this → auto-reject
  upper_threshold: 6.0                # At or above → auto-match
  confidence_threshold: 0.7           # LLM confidence below this → non-match
  features_csv: "../entity-resolution/output/predicted_matches.csv"
```

## Module Descriptions

### dspy_modules.py
Defines `MedicalRecordMatchSignature` (inputs: two medical history summaries; outputs: reasoning, is_match boolean, confidence float) and `MedicalRecordMatcher`, a `ChainOfThought` module that wraps the signature for structured clinical reasoning.

### summarize.py
Converts raw Synthea CSVs into a structured text summary per patient. Conditions and medications are listed in full (compact and highly discriminating). Observations are aggregated to key vitals (height, weight, BMI, blood pressure, A1c, glucose, cholesterol). Encounters are summarized by type and count with the 3 most recent listed. Designed for MedGemma 4B's limited context window.

### optimize.py
Loads gray zone pairs from the classical pipeline, joins with ground truth for labels, generates medical history summaries, and runs MIPROv2 to optimize the `MedicalRecordMatcher`. Supports a separate prompt model (e.g. Claude Sonnet) for generating instruction candidates while the task model (MedGemma) handles evaluation. Falls back to the task model if no API key is set. Splits 75/25 train/val and reports validation accuracy. Saves the optimized program to `data/dspy/optimized_program.json`.

### classify.py
The full hybrid classification pipeline. Loads classical pipeline scored pairs, splits by threshold into auto-reject / auto-match / gray zone, runs the DSPy matcher on gray zone pairs, merges all decisions, evaluates against ground truth, and saves predictions + metrics.

### langfuse_setup.py
Initializes Langfuse tracing from config. Sets environment variables and instruments OpenAI/DSPy calls so every LLM invocation is logged. Falls back gracefully if Langfuse is not running or the package is not installed.

### utils.py
Config loading (`load_config`), MedGemma thinking token extraction (`extract_answer`), and path helpers (`get_project_root`, `get_run_dir`).

## How It Works — Medical Record Summarization

Each patient's raw clinical data (potentially hundreds of rows across 10 record types) is condensed into a structured text summary. Example output:

```
=== MEDICAL HISTORY ===

CONDITIONS (active/historical):
- Diabetes mellitus type 2 (onset: 2010-03-15, ongoing)
- Essential hypertension (onset: 2005-11-22, ongoing)
- Chronic kidney disease stage 2 (onset: 2018-06-01, ongoing)

MEDICATIONS (current/past):
- Metformin 500mg (2010-03-15 to present) for Diabetes mellitus type 2
- Lisinopril 10mg (2005-11-22 to present) for Essential hypertension

ENCOUNTERS (summarized):
- 12 ambulatory visits (2005-2025)
- 3 emergency visits (2018-2024)
Recent:
  - 2025-08-15 ambulatory — Diabetes follow-up

KEY OBSERVATIONS:
- Hemoglobin A1c: 7.2% (2025-08-15)
- Body Mass Index: 31.2 kg/m2 (2025-08-15)
- Systolic Blood Pressure: 138 mmHg (2025-08-15)
- Total observations on file: 418

PROCEDURES:
- Hemoglobin A1c measurement (x8)

IMMUNIZATIONS: Influenza (x15), Td (x3)

ALLERGIES: none

IMAGING: none

DEVICES: none

CARE PLANS: Diabetes self management plan
```

Design choices:
- **Observations** are aggregated to key vitals and labs only (raw data averages 418 rows per patient)
- **Encounters** are aggregated by type with counts and date ranges (raw data averages 35 rows)
- **Conditions and medications** are listed in full — they're compact and highly discriminating for identity matching
- **All 10 record types** are included; DSPy optimization determines which sections contribute most

## Promptfoo Testing

The Promptfoo test suite evaluates prompt quality across models and prompt variants.

### Test structure

Tests are defined in `config/promptfoo/datasets/gray_zone_tests.yaml`. Each test case provides two medical history summaries and assertions about the expected output (match/non-match). Cases cover:

- **True matches**: Same conditions, overlapping medications, consistent timelines
- **True non-matches**: Different conditions, different medications, different onset dates
- **Edge cases**: Sparse records (no history at one facility), near-miss medication overlap

### Running evaluations

```bash
cd llm-entity-resolution/config/promptfoo
promptfoo eval -j 1          # Run all tests sequentially
promptfoo view                # Open results dashboard in browser
```

### Adding new test cases

Add entries to `gray_zone_tests.yaml` following the existing format:

```yaml
- vars:
    medical_history_a: |
      === MEDICAL HISTORY ===
      CONDITIONS: ...
    medical_history_b: |
      === MEDICAL HISTORY ===
      CONDITIONS: ...
  assert:
    - type: icontains
      value: "true"  # or "false" for non-match
```

## Resource Estimates

| Resource | Estimate |
|----------|----------|
| RAM (Ollama + MedGemma Q4) | ~3-4 GB |
| RAM (Ollama + MedGemma full) | ~8-10 GB |
| Optimization time (MIPROv2, ~50 examples) | ~30-60 min |
| Classification time per gray zone pair | ~1-3 sec (Q4-fast) |
| Langfuse (Docker) | ~500 MB RAM |
