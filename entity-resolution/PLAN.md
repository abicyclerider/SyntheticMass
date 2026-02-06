# Entity Resolution Implementation Plan

## Progress Update (as of 2026-02-06)

### Completed
- [x] **Phase 1:** Exploration notebook created (`analysis/notebooks/entity_resolution_exploration.ipynb`)
- [x] **Phase 2:** Framework setup complete (directory structure, requirements.txt, README.md)
- [x] **Phase 3:** Core implementation complete (all 7 modules in `entity-resolution/src/`)
- [x] Updated to use correct data run (`run_20260203_071928` with stripped names)
- [x] Installed dependencies in analysis venv
- [x] **Phase 4 (exploration):** Blocking strategy resolved — aggressive 4-pass union achieves 100% recall
- [x] Similarity distributions, field reliability, error patterns analyzed
- [x] Tiered classification strategy designed (auto-reject / gray zone / auto-match)

### In Progress
- [ ] **Phase 4 (implementation):** Update pipeline modules to use exploration findings
  - Update `blocking.py` with aggressive 4-pass union strategy
  - Replace exact birthdate/SSN matching with fuzzy comparisons in `comparison.py`
  - Implement tiered classification in `classification.py`
  - Update `matching_config.yaml` with tuned parameters

### Not Started
- [ ] **Phase 5:** LLM-based matcher for gray zone pairs (~5.9% of candidates)

### Key Files Created
- `entity-resolution/src/data_loader.py`
- `entity-resolution/src/blocking.py`
- `entity-resolution/src/comparison.py`
- `entity-resolution/src/classification.py`
- `entity-resolution/src/golden_record.py`
- `entity-resolution/src/evaluation.py`
- `entity-resolution/src/pipeline.py`
- `entity-resolution/run_pipeline.py`
- `entity-resolution/config/matching_config.yaml`
- `entity-resolution/requirements.txt`
- `entity-resolution/README.md`
- `analysis/notebooks/entity_resolution_exploration.ipynb`

---

## Exploration Findings

Full analysis in `analysis/notebooks/entity_resolution_exploration.ipynb`.

### Blocking Strategy

**Recommended: Aggressive 4-pass union**

| Pass | Method | New Pairs | New True Pairs | Cumulative Recall |
|------|--------|-----------|----------------|-------------------|
| 1 | LAST + STATE (exact) | 2,059 | 857 | 76.4% |
| 2 | ZIP + BIRTH_YEAR (exact) | +1,211 | +221 | 96.2% |
| 3 | Sorted neighborhood LAST (w=7) | +9,745 | +21 | 98.0% |
| 4 | Sorted neighborhood FIRST (w=5) | +4,984 | +22 | 100.0% |

- **Total:** 17,999 candidate pairs, 100% recall, 97.6% reduction rate
- Passes 1-2 do the heavy lifting (96.2% recall from just 3,270 pairs); Passes 3-4 are safety nets

### Field Reliability (for true matches)

| Field | High Similarity Rate | Notes |
|-------|---------------------|-------|
| zip_match | 100% | Always matches for true pairs |
| state_match | 100% | Single state in dataset |
| city_sim | 98.4% | Very reliable |
| first_name_sim | 96.8% | Occasionally affected by typos/nicknames |
| last_name_sim | 89.9% | Affected by typos and maiden name usage |
| address_sim | 76.4% | Abbreviations, whitespace, apt format variations |
| ssn_match | 66.5% | Digit errors and transpositions (edit distance mean 1.7) |
| birthdate_match | 51.3% | date_off_by_one is most common error type |

### Error Patterns
- **Birthdate:** Only 52% exact match; 10% off by 1 day; 38% off by >2 days. Must use fuzzy date comparison.
- **SSN:** 66.5% exact match; mismatches have low edit distance (mean 1.7). Must use fuzzy SSN comparison.
- **Names:** Typos (203), capitalization (91), maiden name usage (60), nicknames (3). Jaro-Winkler handles most.

### Classification Strategy

**Recommended: Tiered approach with LLM fallback**

Combined similarity score (sum of 8 features, max 8.0):
- **Auto-reject (score < 4.0):** ~15,905 pairs — 0 true matches missed
- **Gray zone (4.0 - 6.0):** ~1,063 pairs (5.9%) — 90 true matches + 973 non-matches → send to LLM
- **Auto-match (score >= 6.0):** ~1,031 pairs — 0 false positives

Without LLM, best single-threshold F1 = 0.983 at threshold 5.60.

---

## Overview
Build an entity resolution system for synthetic medical records using Python Record Linkage Toolkit, with architecture designed for future LLM extension to compare medical histories on difficult matches.

## Context

**Current State:**
- 571 unique patients distributed across 5 facilities
- 1,228 patient-facility records total (avg 2.15 facilities per patient)
- 1,121 true matching pairs
- 42.6% of patients have demographic errors (1,371 total errors across records)
- Ground truth available at `/output/augmented/run_20260203_071928/metadata/ground_truth.csv`
- Error log at `/output/augmented/run_20260203_071928/metadata/error_log.jsonl`

**Error Profile:**
- BIRTHDATE: 357 errors (date_off_by_one)
- ADDRESS: 284 errors (abbreviations, whitespace, apartment format variations)
- SSN: 233 errors (digit errors, transpositions, format variations)
- LAST: 182 errors (typos, capitalization, maiden name usage)
- MAIDEN: 162 errors
- FIRST: 123 errors (typos, capitalization, nicknames)
- CITY: 30 errors

**Goal:**
Create "golden records" by matching patient records across facilities using only demographic fields (not UUIDs), validated against ground truth.

---

## Phase 1: Data Exploration Notebook

**Objective:** Create exploration notebook to understand data characteristics and inform entity resolution design decisions.

### Step 1.1: Create Entity Resolution Exploration Notebook

**File:** `/Users/alex/repos/Kaggle/SyntheticMass/analysis/notebooks/entity_resolution_exploration.ipynb`

**Notebook Sections:**

1. **Setup & Configuration**
   - Import libraries (pandas, recordlinkage, matplotlib)
   - Load ground truth and facility patient data
   - Set run ID: `run_20260203_071928`

2. **Blocking Strategy Analysis**
   - Analyze cardinality of potential blocking keys (first name, last name, ZIP, state)
   - Calculate candidate pair volumes with different blocking strategies
   - Measure blocking recall (% of true matches that pass blocking)
   - **Key Question:** Which blocking strategy balances pair reduction vs match recall?

3. **String Similarity Distributions**
   - Generate candidate pairs using aggressive 4-pass blocking (same as pipeline)
   - Calculate Jaro-Winkler scores for name/address/city fields
   - Compare similarity distributions: true matches vs non-matches
   - **Finding:** No single field cleanly separates matches from non-matches; combined scoring required

4. **Field Reliability Analysis**
   - For each demographic field, measure error frequency
   - Identify most/least reliable fields for matching
   - Analyze error type impact on similarity scores
   - **Key Question:** Which fields should be weighted higher in classification?

5. **Date Error Patterns**
   - Analyze BIRTHDATE error distribution (off-by-one pattern)
   - Test date comparison methods (exact, day tolerance, year-month only)
   - **Key Question:** Should we use fuzzy date matching or exact matching?

6. **SSN Error Analysis**
   - Identify SSN error types (transposition, digit errors, format variations)
   - Test edit distance thresholds for SSN matching
   - **Key Question:** Is SSN reliable enough to be a strong matching signal?

7. **Ground Truth Pair Generation**
   - Create labeled dataset of record pairs (match=1, non-match=0)
   - Threshold tuning and gray zone analysis
   - **Output:** Labeled pairs saved to `analysis/output/labeled_pairs.csv`

### Step 1.2: Document Findings

Key findings documented in notebook summary and this plan's Exploration Findings section:
- Recommended blocking strategy: aggressive 4-pass union (100% recall)
- Per-field thresholds are not viable; combined scoring with tiered classification instead
- Field reliability ranking (ZIP/state best, birthdate/SSN worst)
- Expected F1 > 0.98 with tiered approach

---

## Phase 2: Entity Resolution Framework Setup

**Objective:** Set up project structure and dependencies for entity resolution implementation.

### Step 2.1: Create Entity Resolution Directory

**Directory Structure:**
```
SyntheticMass/
├── augmentation/               # (existing) Data generation & error injection
├── analysis/                   # (existing) Exploratory notebooks
│   └── notebooks/
│       ├── augmentation_eda.ipynb                    # (existing)
│       ├── entity_resolution_exploration.ipynb       # (Phase 1)
│       └── entity_resolution_evaluation.ipynb        # (Phase 4)
├── entity-resolution/          # (new) Entity resolution implementation
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Load facility CSVs and ground truth
│   │   ├── blocking.py              # Blocking/indexing strategies
│   │   ├── comparison.py            # Field comparison functions
│   │   ├── classification.py        # Match/non-match classifier
│   │   ├── golden_record.py         # Merge matched records
│   │   ├── evaluation.py            # Precision/recall/F1 metrics
│   │   └── pipeline.py              # End-to-end workflow orchestration
│   ├── config/
│   │   └── matching_config.yaml     # Configurable parameters
│   ├── output/                      # Generated results
│   │   ├── golden_records.csv
│   │   ├── predicted_matches.csv
│   │   └── evaluation_metrics.json
│   ├── tests/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── test_blocking.py
│   │   ├── test_comparison.py
│   │   └── test_classification.py
│   ├── requirements.txt             # ER-specific dependencies
│   ├── README.md                    # Documentation
│   └── run_pipeline.py              # CLI entry point
├── output/                     # (existing) Augmentation outputs
└── synthea-runner/             # (existing) Base patient generation
```

### Step 2.2: Create Requirements File

**File:** `/Users/alex/repos/Kaggle/SyntheticMass/entity-resolution/requirements.txt`

```
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0

# Record linkage
recordlinkage>=0.15

# Machine learning
scikit-learn>=1.3.0

# String similarity
jellyfish>=1.0.0
python-Levenshtein>=0.21.0

# Utilities
click>=8.1.0
rich>=13.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

Install with: `cd entity-resolution && pip install -r requirements.txt`

---

## Phase 3: Core Implementation

**Objective:** Implement entity resolution pipeline using Python Record Linkage Toolkit.

### Step 3.0: Create README

**File:** `entity-resolution/README.md`

**Contents:**
- Overview of entity resolution system
- Installation instructions
- Usage examples (CLI and programmatic)
- Configuration options
- Module descriptions
- LLM extension architecture notes

### Step 3.1: Data Loader Module

**File:** `entity-resolution/src/data_loader.py`

**Responsibilities:**
- Load patient CSVs from all facilities into single DataFrame
- Add `facility_id` column to track source
- Load ground truth for validation
- Derive BIRTH_YEAR column from BIRTHDATE (needed for blocking)
- Basic cleaning (strip whitespace, normalize case)

**Key Functions:**
```python
def load_facility_patients(run_dir: str) -> pd.DataFrame
def load_ground_truth(run_dir: str) -> pd.DataFrame
def prepare_for_matching(df: pd.DataFrame) -> pd.DataFrame
```

### Step 3.2: Blocking Module

**File:** `entity-resolution/src/blocking.py`

**Responsibilities:**
- Implement multiple blocking strategies based on Phase 1 findings
- Reduce candidate pairs from O(n²) to manageable size
- Track blocking statistics (pairs generated, reduction rate)

**Primary Strategy: Aggressive 4-pass union** (100% recall, 17,999 pairs, 97.6% reduction)
1. Block on LAST + STATE (exact) — 76.4% recall baseline
2. Block on ZIP + BIRTH_YEAR (exact) — catches name-error cases, brings to 96.2%
3. Sorted neighborhood on LAST (w=7) — catches surname typos, brings to 98.0%
4. Sorted neighborhood on FIRST (w=5) — catches remaining edge cases, brings to 100%

**Key Functions:**
```python
def create_candidate_pairs(df: pd.DataFrame, strategy: str) -> pd.MultiIndex
def evaluate_blocking_recall(pairs: pd.MultiIndex, ground_truth: pd.DataFrame) -> float
```

### Step 3.3: Comparison Module

**File:** `entity-resolution/src/comparison.py`

**Responsibilities:**
- Calculate similarity scores for each field pair
- Use findings from Phase 1 to select appropriate methods
- Handle missing values gracefully

**Comparison Methods (informed by exploration findings):**
- **Names (FIRST, LAST):** Jaro-Winkler similarity — handles typos, capitalization, nicknames
- **SSN:** Exact match currently, but must switch to fuzzy (edit distance ≤ 2) — only 66.5% exact match rate for true pairs
- **BIRTHDATE:** Exact match currently, but must switch to fuzzy (year match + day diff) — only 51.3% exact match rate for true pairs
- **ADDRESS:** Jaro-Winkler — handles abbreviations, whitespace, apartment format variations
- **CITY:** Jaro-Winkler (98.4% reliable, occasional errors)
- **STATE, ZIP:** Exact match (100% reliable for true pairs)

**Key Functions:**
```python
def build_comparison_features(pairs: pd.MultiIndex, df: pd.DataFrame) -> pd.DataFrame
def create_custom_comparator() -> recordlinkage.Compare
```

### Step 3.4: Classification Module

**File:** `entity-resolution/src/classification.py`

**Responsibilities:**
- Classify candidate pairs as match/non-match
- Support multiple classification approaches
- Design with LLM extension point for future enhancement

**Classification Approach: Tiered with LLM fallback**

Based on exploration findings (sum of 8 Jaro-Winkler/exact features, max score 8.0):

1. **Auto-reject (score < 4.0):** ~15,905 pairs — 0 false negatives
2. **Gray zone (4.0 - 6.0):** ~1,063 pairs (5.9%) — send to LLM matcher (Phase 5)
3. **Auto-match (score >= 6.0):** ~1,031 pairs — 0 false positives

Fallback without LLM: single threshold at 5.60 gives F1 = 0.983.

**Key Functions:**
```python
def classify_tiered(features: pd.DataFrame, low: float, high: float) -> pd.Series
    # Returns: 'match', 'non_match', or 'uncertain'
def classify_with_llm_fallback(features: pd.DataFrame, medical_data: pd.DataFrame) -> pd.Series
    # Resolves 'uncertain' pairs via LLM
```

### Step 3.5: Golden Record Module

**File:** `entity-resolution/src/golden_record.py`

**Responsibilities:**
- Merge matched records into single consolidated record per patient
- Resolve conflicts across facilities (which value to trust?)
- Track provenance (which facilities contributed to golden record)

**Conflict Resolution Strategy:**
1. **Most frequent value** (democratic voting)
2. **Most recent encounter** (temporal preference)
3. **Least errors** (trust facilities with historically fewer errors)
4. **Field-specific rules** (e.g., prefer non-abbreviated addresses)

**Key Functions:**
```python
def create_golden_records(matches: pd.DataFrame, patient_data: pd.DataFrame) -> pd.DataFrame
def resolve_field_conflict(values: list, strategy: str) -> Any
def merge_medical_histories(encounter_dfs: list) -> pd.DataFrame
```

### Step 3.6: Evaluation Module

**File:** `entity-resolution/src/evaluation.py`

**Responsibilities:**
- Compare predicted matches against ground truth
- Calculate precision, recall, F1-score
- Generate confusion matrix and error analysis
- Identify hard-to-match patient clusters

**Key Functions:**
```python
def evaluate_matches(predicted_pairs: pd.MultiIndex, ground_truth: pd.DataFrame) -> dict
def calculate_metrics(tp: int, fp: int, fn: int) -> dict
def error_analysis(false_positives: pd.DataFrame, false_negatives: pd.DataFrame) -> pd.DataFrame
```

### Step 3.7: Pipeline Orchestration

**File:** `entity-resolution/src/pipeline.py`

**Responsibilities:**
- Coordinate end-to-end workflow
- Load data → Block → Compare → Classify → Merge → Evaluate
- Log performance metrics at each stage
- Support configuration via YAML file

**Key Functions:**
```python
def run_entity_resolution_pipeline(config: dict) -> dict
def load_config(config_path: str) -> dict
def save_results(golden_records: pd.DataFrame, metrics: dict, output_dir: str)
```

### Step 3.8: CLI Entry Point

**File:** `entity-resolution/run_pipeline.py`

**Responsibilities:**
- Command-line interface for running entity resolution pipeline
- Load configuration from YAML file
- Display progress and results
- Save outputs to entity-resolution/output/

**Key Features:**
```python
import click
from src.pipeline import run_entity_resolution_pipeline, load_config

@click.command()
@click.option('--config', default='config/matching_config.yaml', help='Path to config file')
@click.option('--output-dir', default='output/', help='Output directory for results')
@click.option('--verbose', is_flag=True, help='Verbose output')
def main(config, output_dir, verbose):
    """Run entity resolution pipeline on augmented patient data."""
    config_dict = load_config(config)
    results = run_entity_resolution_pipeline(config_dict, output_dir, verbose)

    print(f"\nEntity Resolution Complete!")
    print(f"  Golden records: {results['num_golden_records']}")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1 Score: {results['f1_score']:.3f}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()
```

### Step 3.9: Configuration File

**File:** `entity-resolution/config/matching_config.yaml`

```yaml
# Entity Resolution Configuration
run_id: "run_20260203_071928"
base_dir: "/Users/alex/repos/Kaggle/SyntheticMass"

# Blocking strategy — aggressive 4-pass union
blocking:
  strategy: "aggressive_multipass"
  passes:
    - type: block
      fields: [LAST, STATE]
    - type: block
      fields: [ZIP, BIRTH_YEAR]
    - type: sorted_neighbourhood
      field: LAST
      window: 7
    - type: sorted_neighbourhood
      field: FIRST
      window: 5

# Comparison methods (use fuzzy matching — exact matching misses too many true pairs)
comparison:
  first_name: {method: jarowinkler}
  last_name: {method: jarowinkler}
  address: {method: jarowinkler}
  city: {method: jarowinkler}
  state: {method: exact}
  zip: {method: exact}
  ssn: {method: exact}           # TODO: replace with fuzzy (edit distance ≤ 2)
  birthdate: {method: exact}     # TODO: replace with fuzzy (year match + day diff)

# Classification — tiered approach
classification:
  method: "tiered"
  auto_reject_threshold: 4.0     # score < 4.0 → non-match (0 false negatives)
  auto_match_threshold: 6.0      # score >= 6.0 → match (0 false positives)
  gray_zone: "llm"               # 4.0-6.0 → send to LLM (~5.9% of pairs)
  # Fallback single threshold if no LLM available:
  single_threshold: 5.60         # F1 = 0.983

# Golden record creation
golden_record:
  conflict_resolution: "most_frequent"

# Output
output:
  golden_records_csv: "golden_records.csv"
  matches_csv: "predicted_matches.csv"
  metrics_json: "evaluation_metrics.json"
```

---

## Phase 4: Evaluation & Iteration

**Objective:** Validate entity resolution performance and iterate on design.

### Step 4.1: Create Evaluation Notebook

**File:** `/Users/alex/repos/Kaggle/SyntheticMass/analysis/notebooks/entity_resolution_evaluation.ipynb`

**Notebook Sections:**
1. Run entity resolution pipeline with different configurations
2. Compare precision/recall/F1 across approaches
3. Error analysis: Which patient types are hard to match?
4. Visualization: Confusion matrix, similarity distributions
5. Golden records quality assessment

### Step 4.2: Iterate on Parameters

Based on evaluation results:
- Blocking is solved (100% recall) — iterate on comparison and classification
- Replace exact SSN/birthdate matching with fuzzy comparisons to improve feature quality
- Tune gray zone boundaries (currently 4.0-6.0) based on pipeline results
- Refine conflict resolution rules for golden records

### Step 4.3: Document Final Performance

Create summary report:
- Final precision, recall, F1 scores
- Comparison to baseline approaches
- Identified limitations and edge cases
- Recommendations for LLM enhancement

---

## Phase 5: LLM Extension Architecture (Future Work)

**Objective:** Build LLM-based matcher for gray zone pairs.

### Architecture Design

**When to invoke LLM:**
- Demographic similarity score in gray zone (4.0 - 6.0)
- ~1,063 pairs (5.9% of candidates) containing 90 true matches and 973 non-matches
- Auto-reject and auto-match tiers handle the other 94.1% with zero errors

**LLM Input:**
```
Patient A:
- Demographics: [similarity scores]
- Medical history: [encounters, conditions, medications]

Patient B:
- Demographics: [similarity scores]
- Medical history: [encounters, conditions, medications]

Question: Based on medical history patterns, are these records for the same patient?
Consider: treatment continuity, chronic conditions, medication patterns, provider networks.
```

**LLM Output:**
- Match confidence score (0-1)
- Reasoning (which medical patterns support/contradict match)

**Implementation Notes:**
- Add medical history loading in entity-resolution/src/data_loader.py
- Create prompt templates in new module: `entity-resolution/src/llm_comparison.py`
- Integrate with entity-resolution/src/classification.py via `classify_with_llm_fallback()`
- Cache LLM results to avoid redundant API calls
- Track LLM invocation costs and accuracy improvement

---

## Critical Files

**Created (need updating to match exploration findings):**
- `entity-resolution/src/blocking.py` — needs aggressive 4-pass union strategy
- `entity-resolution/src/comparison.py` — needs fuzzy SSN/birthdate comparisons
- `entity-resolution/src/classification.py` — needs tiered classification
- `entity-resolution/config/matching_config.yaml` — needs updated parameters
- `entity-resolution/src/data_loader.py` — needs BIRTH_YEAR derivation
- `entity-resolution/src/golden_record.py`
- `entity-resolution/src/evaluation.py`
- `entity-resolution/src/pipeline.py`
- `entity-resolution/run_pipeline.py`
- `analysis/notebooks/entity_resolution_exploration.ipynb`

**To Create:**
- `entity-resolution/src/llm_comparison.py` — Phase 5, gray zone LLM matcher
- `analysis/notebooks/entity_resolution_evaluation.ipynb` — Phase 4 evaluation

**Reference (Read-Only):**
- `output/augmented/run_20260203_071928/metadata/ground_truth.csv`
- `output/augmented/run_20260203_071928/metadata/error_log.jsonl`
- `output/augmented/run_20260203_071928/facilities/facility_*/patients.csv`

---

## Verification & Testing

### End-to-End Test:
```bash
# 1. Run exploration notebook (in analysis folder)
cd /Users/alex/repos/Kaggle/SyntheticMass/analysis
jupyter notebook notebooks/entity_resolution_exploration.ipynb
# Execute all cells, verify findings documented

# 2. Set up entity resolution environment
cd /Users/alex/repos/Kaggle/SyntheticMass/entity-resolution
pip install -r requirements.txt

# 3. Run entity resolution pipeline
python run_pipeline.py --config config/matching_config.yaml

# Alternative: Run programmatically
python -c "from src.pipeline import run_entity_resolution_pipeline;
           from src.pipeline import load_config;
           config = load_config('config/matching_config.yaml');
           results = run_entity_resolution_pipeline(config);
           print(f'F1 Score: {results[\"f1_score\"]}')"

# 4. Run evaluation notebook (in analysis folder)
cd /Users/alex/repos/Kaggle/SyntheticMass/analysis
jupyter notebook notebooks/entity_resolution_evaluation.ipynb
# Execute all cells, verify metrics > 0.98 F1
```

### Success Criteria:
- [x] Exploration notebook runs without errors
- [x] All 7 analysis sections complete with findings
- [x] Blocking achieves 100% recall (1,121/1,121 true pairs) with 17,999 candidates
- [ ] Entity resolution pipeline processes all 1,228 records
- [ ] F1 score > 0.98 on ground truth validation (0.983 demonstrated in exploration)
- [ ] Golden records CSV generated with 571 unique patients
- [ ] Evaluation metrics JSON contains precision/recall/F1
- [ ] LLM matcher handles gray zone pairs (5.9% of candidates)

### Unit Testing (Optional but Recommended):
```bash
cd /Users/alex/repos/Kaggle/SyntheticMass/entity-resolution

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Notes

- Phase 1 exploration complete — all design decisions are now data-driven
- Tiered classification (auto-reject / gray zone / auto-match) replaces simple threshold approach
- LLM matcher is Phase 5 — only needed for ~5.9% of candidate pairs
- Ground truth validation is critical — use it at every stage
- Fuzzy SSN and birthdate comparison are the highest-impact improvements remaining
- The notebook uses uppercase field names (FIRST, LAST, etc.) — pipeline should match
