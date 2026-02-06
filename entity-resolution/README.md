# Entity Resolution System

## Overview

This entity resolution system matches patient records across multiple healthcare facilities to create "golden records" - consolidated patient identities that combine information from multiple sources while handling demographic errors and variations.

The system uses the Python Record Linkage Toolkit to perform probabilistic record linkage based on demographic fields (names, addresses, dates of birth, SSNs) without relying on unique identifiers like UUIDs.

**Key Features:**
- Multiple blocking strategies to reduce candidate pair space
- Configurable string similarity comparisons for fuzzy matching
- Support for rule-based, probabilistic, and hybrid classification
- Ground truth validation and performance metrics
- Designed with extension points for future LLM-based medical history comparison

## Architecture

The system is designed with modularity and extensibility in mind:

```
src/
├── data_loader.py       # Load facility CSVs and ground truth
├── blocking.py          # Indexing strategies to reduce pair space
├── comparison.py        # Field-by-field similarity calculations
├── classification.py    # Match/non-match decision logic
├── golden_record.py     # Merge matched records with conflict resolution
├── evaluation.py        # Precision/recall metrics against ground truth
└── pipeline.py          # End-to-end workflow orchestration
```

### LLM Extension Architecture

The classification module is designed with an extension point for LLM-based matching:

1. **Deterministic rules** handle high-confidence matches (exact SSN + DOB)
2. **Probabilistic models** handle medium-confidence matches
3. **LLM fallback** (future) for borderline cases where medical history comparison would help

The LLM would receive:
- Demographic similarity scores
- Medical encounter histories for both candidate records
- Prompt asking: "Based on treatment continuity, chronic conditions, and medication patterns, are these the same patient?"

This hierarchical approach keeps costs low while leveraging LLM reasoning on difficult edge cases.

## Installation

```bash
cd entity-resolution
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

Run the complete entity resolution pipeline:

```bash
python run_pipeline.py --config config/matching_config.yaml --output-dir output/
```

Options:
- `--config`: Path to YAML configuration file (default: `config/matching_config.yaml`)
- `--output-dir`: Directory for output files (default: `output/`)
- `--verbose`: Enable verbose logging

### Programmatic Usage

```python
from src.pipeline import run_entity_resolution_pipeline, load_config

# Load configuration
config = load_config('config/matching_config.yaml')

# Run pipeline
results = run_entity_resolution_pipeline(config)

# Access results
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1 Score: {results['f1_score']:.3f}")
print(f"Golden records: {results['num_golden_records']}")
```

### Module Usage Examples

**Data Loading:**
```python
from src.data_loader import load_facility_patients, load_ground_truth

# Load all facility patient records
patients_df = load_facility_patients(run_dir)

# Load ground truth for validation
ground_truth_df = load_ground_truth(run_dir)
```

**Blocking:**
```python
from src.blocking import create_candidate_pairs

# Generate candidate pairs using blocking strategy
pairs = create_candidate_pairs(patients_df, strategy='lastname_state')
```

**Comparison:**
```python
from src.comparison import build_comparison_features

# Calculate similarity scores for all candidate pairs
features_df = build_comparison_features(pairs, patients_df)
```

**Classification:**
```python
from src.classification import classify_threshold

# Classify pairs as matches/non-matches
matches = classify_threshold(features_df, threshold=3.5)
```

**Golden Records:**
```python
from src.golden_record import create_golden_records

# Merge matched records into consolidated golden records
golden_records_df = create_golden_records(matches, patients_df)
```

**Evaluation:**
```python
from src.evaluation import evaluate_matches

# Compare predictions against ground truth
metrics = evaluate_matches(predicted_pairs, ground_truth_df)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

## Configuration

The system is configured via YAML file (`config/matching_config.yaml`):

```yaml
# Data source
run_id: "run_20260202_122731"
base_dir: "/Users/alex/repos/Kaggle/SyntheticMass"

# Blocking strategy
blocking:
  strategy: "lastname_state"  # Options: lastname_state, sorted_neighborhood, zip_birthyear
  sorted_neighborhood_window: 5

# Comparison thresholds
comparison:
  first_name_threshold: 0.85    # Jaro-Winkler similarity
  last_name_threshold: 0.85
  address_threshold: 0.80
  ssn_exact_match: true
  birthdate_tolerance_days: 1

# Classification method
classification:
  method: "threshold"  # Options: threshold, logistic_regression, llm_fallback
  threshold: 3.5

# Golden record creation
golden_record:
  conflict_resolution: "most_frequent"  # Options: most_frequent, most_recent
```

## Output Files

The pipeline generates three output files:

1. **golden_records.csv**: Consolidated patient records (one per unique patient)
2. **predicted_matches.csv**: All matched record pairs with similarity scores
3. **evaluation_metrics.json**: Performance metrics (precision, recall, F1)

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Analysis Notebooks

See the `analysis/notebooks/` directory for:

- **entity_resolution_exploration.ipynb**: Data exploration and parameter tuning
- **entity_resolution_evaluation.ipynb**: Performance analysis and error cases

## Module Descriptions

### data_loader.py
Loads patient records from multiple facility CSVs and combines them into a single DataFrame with facility tracking. Performs basic data cleaning and standardization. Also loads ground truth for validation.

### blocking.py
Implements blocking (indexing) strategies to reduce the O(n²) comparison space. Strategies include blocking on last name + state, sorted neighborhood, and multi-pass approaches. Tracks blocking recall to ensure true matches aren't excluded.

### comparison.py
Calculates field-by-field similarity scores for candidate pairs using appropriate methods:
- Jaro-Winkler for names and addresses (handles typos)
- Exact or fuzzy matching for SSN
- Date comparison with configurable tolerance for birthdates
- Exact matching for structured fields (ZIP, state)

### classification.py
Classifies candidate pairs as matches or non-matches. Supports multiple approaches:
- **Threshold-based**: Sum similarity scores and apply cutoff
- **Probabilistic**: Logistic regression trained on labeled pairs
- **LLM fallback**: Extension point for medical history comparison (future)

### golden_record.py
Merges matched records into consolidated golden records. Handles conflicts when different facilities have different values for the same patient using configurable strategies:
- Most frequent value (democratic voting)
- Most recent encounter (temporal preference)
- Field-specific rules

### evaluation.py
Compares predicted matches against ground truth to calculate precision, recall, and F1 scores. Provides error analysis to identify false positives and false negatives.

### pipeline.py
Orchestrates the end-to-end workflow: load data → block → compare → classify → merge → evaluate. Handles configuration loading, progress tracking, and result saving.

## Performance Expectations

Based on the augmented dataset (571 unique patients, 1228 records, 42.6% error rate):

- **Target F1 Score**: > 0.85
- **Blocking Reduction**: ~99% of candidate pairs eliminated
- **Golden Records**: 571 (one per unique patient)

## Future Enhancements

1. **LLM-based medical history comparison** for borderline cases
2. **Active learning** to improve classifier with user feedback
3. **Real-time matching** for streaming patient records
4. **Multi-modal matching** incorporating clinical notes and imaging metadata
5. **Temporal entity resolution** tracking patient identity changes over time

## Contributing

When adding new features:
1. Add unit tests in `tests/`
2. Update configuration schema if needed
3. Document new parameters in this README
4. Run full test suite before committing

## License

Part of the SyntheticMass Kaggle project.
