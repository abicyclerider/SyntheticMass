# Shared Utilities

Common data loading, evaluation, and summarization used across the pipeline.

## Modules

### data_loader.py

Load patient records from multi-facility augmented output into a single DataFrame.

- **`load_facility_patients(run_dir)`** — Loads all facility patient parquet files, adds `facility_id` column, standardizes columns automatically.
- **`standardize_columns(df)`** — Normalizes Synthea column names (e.g., `FIRST` to `first_name`), title-cases names, strips SSN dashes, zero-pads ZIP codes, parses birthdates.
- **`create_record_id(df)`** — Creates composite identifiers (`facility_001_uuid-abc`) for unique cross-facility record tracking.

### ground_truth.py

Load ground truth mappings and generate true matching pairs for evaluation.

- **`load_ground_truth(run_dir)`** — Loads `metadata/ground_truth.parquet`, normalizes facility IDs to `facility_NNN` format.
- **`generate_true_pairs_from_ground_truth(ground_truth)`** — Generates all C(n,2) record ID pairs for each true patient. Returns a set of sorted tuples for order-independent comparison.
- **`add_record_ids_to_ground_truth(ground_truth_df, patients_df)`** — Enriches ground truth with composite record IDs via join on facility + patient UUID.

### medical_records.py

Load clinical record types (conditions, medications, observations, etc.) from facility parquet files.

- **`load_medical_records(run_dir, record_types=None)`** — Loads all facility data for specified record types (default: 11 clinical types, excluding financial). Returns a dict keyed by record type name.
- **`get_patient_records(patient_id, facility_id, medical_records)`** — Filters to a single patient at a single facility. Used by the summarizer.
- **`CLINICAL_RECORD_TYPES`** — List of 11 record types: encounters, conditions, medications, observations, procedures, immunizations, allergies, careplans, imaging_studies, devices, supplies.

### summarize.py

Strategy D summarizer — generates structured, diff-friendly clinical summaries for pairwise LLM comparison.

- **`summarize_diff_friendly(patient_id, facility_id, medical_records)`** — Fetches patient records and produces a structured summary.
- **`summarize_diff_friendly_from_records(records)`** — Core summarization logic. Produces sections for conditions (grouped by onset year), medications (with date ranges), allergies, observations (latest 2 values per vital sign), and procedures (grouped by description).
- **`INSTRUCTION`** — Prompt template for the LLM classifier with `{summary_a}` and `{summary_b}` placeholders.

Clinical summaries use year-grouped formatting for easy visual comparison. Only the most recent 2 observations per metric are included to keep summaries concise (~800 tokens).

### evaluation.py

Pair-level evaluation metrics for entity resolution.

- **`calculate_confusion_matrix(predicted_pairs, true_pairs)`** — Computes TP, FP, FN from sets of pair tuples. Normalizes pair order so (A,B) and (B,A) are treated as identical.
- **`calculate_metrics(tp, fp, fn)`** — Computes precision, recall, and F1 from confusion matrix counts. Handles zero-division gracefully.

## Usage

```python
from shared.data_loader import load_facility_patients, create_record_id
from shared.ground_truth import load_ground_truth, generate_true_pairs_from_ground_truth
from shared.medical_records import load_medical_records
from shared.summarize import summarize_diff_friendly
from shared.evaluation import calculate_confusion_matrix, calculate_metrics

# Load patient data
patients = load_facility_patients("output/augmented/run_001")
patients = create_record_id(patients)

# Load ground truth and generate true pairs
gt = load_ground_truth("output/augmented/run_001")
true_pairs = generate_true_pairs_from_ground_truth(gt)

# Load clinical records and summarize a patient
records = load_medical_records("output/augmented/run_001")
summary = summarize_diff_friendly("uuid-123", "facility_001", records)

# Evaluate predictions
tp, fp, fn = calculate_confusion_matrix(predicted_pairs, true_pairs)
metrics = calculate_metrics(tp, fp, fn)
```
