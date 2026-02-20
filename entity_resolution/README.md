# Entity Resolution

Probabilistic entity resolution using [Splink v4](https://github.com/moj-analytical-services/splink) (Fellegi-Sunter model) with MedGemma-powered gray zone classification.

## Overview

The entity resolution pipeline takes augmented multi-facility patient data and produces a master patient index (golden records). It operates in three tiers:

1. **Auto-match** — Splink assigns match probabilities to candidate pairs. Pairs above 0.99 are accepted without LLM review.
2. **Gray zone** — Pairs between 0.01 and 0.99 are ambiguous on demographics alone. These are enriched with structured clinical summaries and sent to a fine-tuned MedGemma classifier.
3. **Auto-reject** — Pairs below 0.01 are discarded.

Final matches from both tiers are clustered into connected components and merged into golden records with field-level conflict resolution.

## Pipeline Stages

### resolve.py

Main resolution script. Trains a Splink model, predicts match probabilities, splits predictions into tiers, and generates clinical summaries for gray zone pairs.

```bash
python -m entity_resolution.resolve \
    --augmented-dir /data/augmented \
    --output-dir /data/resolved \
    --config entity_resolution/config/matching_config.yaml
```

**Outputs:** `auto_matches.parquet`, `gray_zone_pairs.parquet`, `features.parquet`, `resolve_metrics.json`

### build_golden_records.py

Combines auto-matches with LLM predictions using logit-space fusion, clusters matches via connected components, merges patient records, and evaluates against ground truth.

```bash
python -m entity_resolution.build_golden_records \
    --augmented-dir /data/augmented \
    --auto-matches /data/resolved/auto_matches.parquet \
    --predictions /data/inferred/predictions.parquet \
    --features /data/resolved/features.parquet \
    --output-dir /data/golden_records \
    --config entity_resolution/config/matching_config.yaml
```

**Outputs:** `golden_records.parquet`, `all_matches.parquet`, `evaluation_metrics.json`

## Splink Configuration

**7 comparison fields:**

| Field | Method | Notes |
|-------|--------|-------|
| first_name | Jaro-Winkler (0.92, 0.80) | Term frequency adjusted |
| last_name | Jaro-Winkler (0.92, 0.80) | Term frequency adjusted |
| address | Jaro-Winkler (0.9, 0.7) | |
| city | Jaro-Winkler (0.9, 0.7) | Term frequency adjusted |
| zip | Exact match | |
| ssn | Exact match | |
| birthdate | Date proximity tiers | |

State is excluded — all records are Massachusetts, so it has zero discriminating power.

**7 blocking rules** combine last_name/city, zip/birth_year, first_name/birth_year, SSN, birthdate, first_name/zip, and address to generate candidate pairs.

**Unsupervised EM training** runs 3 sessions with rotating blocking rules to estimate m-probabilities for all comparison fields. Lambda and u-probabilities are estimated separately.

## Gray Zone Fusion

Gray zone predictions combine two signals in log-odds space with a Bayesian prior correction:

```
combined_logit = w_splink * log(p/(1-p)) + w_llm * (llm_logit + prior_correction)
```

The prior correction accounts for the mismatch between the balanced training data (50/50) and the production gray zone (~1% true matches): `correction = logit(gray_zone_prior) - logit(llm_training_prior)`.

- `w_splink` (default 1.0) — weight on Splink demographic signal
- `w_llm` (default 1.5) — weight on LLM clinical signal (LLM more reliable than demographics with heavy errors)
- `threshold` (default -5.5) — combined logit threshold (grid-searched on 1k patient eval)
- `min_splink_probability` (default 0.0) — Splink probability floor veto (currently disabled)
- `llm_training_prior` (default 0.5) — base rate in LLM training data (balanced 50/50)
- `gray_zone_prior` (default 0.01) — estimated true-match rate in the gray zone

## Golden Record Creation

**Clustering:** Matched pairs are assembled into connected components via DFS. If A matches B and B matches C, all three belong to the same patient. Unmatched records become singleton clusters.

**Conflict resolution:** Each cluster's records are merged field by field. The default strategy is democratic voting (most frequent value). Field-specific rules apply domain knowledge:
- **address** — prefer longer (less abbreviated) form
- **SSN** — prefer formatted (XXX-XX-XXXX)
- **names** — prefer title case

**Provenance:** Golden records track contributing facilities and source record IDs.

## Evaluation

**Pair-level:** Precision, recall, and F1 computed against ground truth pairs. Pair order is normalized (A,B == B,A).

**Record-level:** Golden record count compared to true unique patient count. Current pipeline produces 599 golden records from 600 true patients.

## Configuration

Key parameters from [`config/matching_config.yaml`](config/matching_config.yaml):

```yaml
splink:
  predict_threshold: 0.0001      # Minimum probability to keep as candidate
  auto_match_probability: 0.99   # P >= 0.99 → definite match
  auto_reject_probability: 0.0001 # P < 0.0001 → definite non-match

gray_zone:
  w_splink: 1.0                  # Splink logit weight
  w_llm: 1.5                    # LLM logit weight
  threshold: -5.5                # Combined logit threshold (grid-searched)
  min_splink_probability: 0.0    # Floor: disabled (no veto)
  llm_training_prior: 0.5        # Base rate in training data (balanced 50/50)
  gray_zone_prior: 0.01          # Estimated true-match rate in gray zone

golden_record:
  conflict_resolution: "most_frequent"
  include_provenance: true
```

These thresholds are also parameterized in `params.yaml` for DVC pipeline control.

## Usage

Both scripts run inside Docker via DVC (see top-level [`dvc.yaml`](../dvc.yaml)):

```bash
# Run Splink resolution
dvc repro resolve

# Run golden record creation (requires infer stage first)
dvc repro golden_records
```

Or directly without Docker:

```bash
python -m entity_resolution.resolve \
    --augmented-dir output/augmented \
    --output-dir output/resolved \
    --config entity_resolution/config/matching_config.yaml

python -m entity_resolution.build_golden_records \
    --augmented-dir output/augmented \
    --auto-matches output/resolved/auto_matches.parquet \
    --predictions output/inferred/predictions.parquet \
    --features output/resolved/features.parquet \
    --output-dir output/golden_records \
    --config entity_resolution/config/matching_config.yaml
```
