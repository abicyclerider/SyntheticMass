# Quick Start Guide

## Prerequisites

1. Activate virtual environment:
```bash
cd /Users/alex/repos/Kaggle/SyntheticMass
source augmentation/venv/bin/activate
```

## Generate Augmented Data

### Default Configuration (10 facilities, 35% error rate)

```bash
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented
```

### Custom Number of Facilities

```bash
# 5 facilities
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented \
  --facilities 5

# 20 facilities
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented \
  --facilities 20
```

### Custom Error Rate

```bash
# 50% error rate
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented \
  --error-rate 0.50
```

### Dry Run (Test Configuration)

```bash
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented \
  --dry-run
```

## View Results

### Output Structure

```bash
# List facility directories
ls output/augmented/run_*/facilities/

# Check CSV files in a facility
ls output/augmented/run_*/facilities/facility_001/

# View ground truth
head -20 output/augmented/run_*/metadata/ground_truth.csv

# View facilities
cat output/augmented/run_*/metadata/facilities.csv

# View statistics
cat output/augmented/run_*/statistics/augmentation_report.json
```

### Example: Find Same Patient Across Facilities

```bash
# Get a patient UUID from ground truth
patient_uuid=$(head -2 output/augmented/run_*/metadata/ground_truth.csv | tail -1 | cut -d',' -f1)

# Search for this patient in all facilities
grep "$patient_uuid" output/augmented/run_*/facilities/*/patients.csv
```

## Run Tests

```bash
# All tests
pytest augmentation/tests/

# Just unit tests (fast)
pytest augmentation/tests/unit/ -v

# Just integration tests
pytest augmentation/tests/integration/ -v
```

## Common Use Cases

### 1. Generate Data for Entity Resolution Testing

```bash
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented_er_test \
  --facilities 10 \
  --error-rate 0.35 \
  --random-seed 42
```

### 2. Generate Multiple Datasets with Different Seeds

```bash
# Dataset 1
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented_seed1 \
  --random-seed 1

# Dataset 2
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented_seed2 \
  --random-seed 2
```

### 3. Use Custom Configuration File

```bash
# Edit config
cp augmentation/config/default_config.yaml my_config.yaml
# ... edit my_config.yaml ...

# Run with custom config
python -m augmentation.cli.augment \
  --input synthea-runner/output/synthea_raw/csv \
  --output output/augmented \
  --config my_config.yaml
```

## Inspect Output

### Load and Analyze in Python

```python
import pandas as pd

# Load ground truth
gt = pd.read_csv('output/augmented/run_*/metadata/ground_truth.csv')

# Find patients at multiple facilities
multi_facility = gt.groupby('original_patient_uuid').size()
multi_facility = multi_facility[multi_facility > 1]
print(f"{len(multi_facility)} patients appear at multiple facilities")

# Load facility data
facility1_patients = pd.read_csv('output/augmented/run_*/facilities/facility_001/patients.csv')
facility2_patients = pd.read_csv('output/augmented/run_*/facilities/facility_002/patients.csv')

# Find same patient in both facilities
patient_uuid = multi_facility.index[0]
p1 = facility1_patients[facility1_patients['Id'] == patient_uuid]
p2 = facility2_patients[facility2_patients['Id'] == patient_uuid]

# Compare demographics (will differ due to errors)
print("Facility 1:", p1[['FIRST', 'LAST', 'ADDRESS']].values)
print("Facility 2:", p2[['FIRST', 'LAST', 'ADDRESS']].values)
```

### View Error Log

```bash
# Count errors by type
cat output/augmented/run_*/metadata/error_log.jsonl | \
  jq -r '.error_type' | \
  sort | uniq -c | sort -rn

# View errors for specific patient
patient_uuid="..."
cat output/augmented/run_*/metadata/error_log.jsonl | \
  jq "select(.patient_uuid == \"$patient_uuid\")"
```

## Troubleshooting

### "Module not found" error

```bash
# Make sure you're in the right directory
cd /Users/alex/repos/Kaggle/SyntheticMass

# Activate virtual environment
source augmentation/venv/bin/activate
```

### "No such file or directory" for input

```bash
# Check input path exists
ls -la synthea-runner/output/synthea_raw/csv/
```

### Tests failing

```bash
# Reinstall dependencies
pip install -r augmentation/requirements.txt

# Run tests with verbose output
pytest augmentation/tests/ -v -s
```

## Configuration Reference

Edit `augmentation/config/default_config.yaml`:

```yaml
facility_distribution:
  num_facilities: 10              # Total facilities
  facility_count_weights:         # Patient distribution
    1: 0.40  # 40% visit 1 facility
    2: 0.30  # 30% visit 2 facilities
    3: 0.15  # 15% visit 3 facilities
    4: 0.10  # 10% visit 4 facilities
    5: 0.05  # 5% visit 5 facilities
  primary_facility_weight: 0.60   # 60% encounters at primary

error_injection:
  global_error_rate: 0.35         # 35% of records have errors
  multiple_errors_probability: 0.20  # 20% have multiple errors
  error_type_weights:             # Relative weights
    name_variation: 0.30
    address_error: 0.25
    date_variation: 0.15
    ssn_error: 0.10
    formatting_error: 0.20

random_seed: 42                   # For reproducibility
```

## Next Steps

1. **Generate full dataset** for your use case
2. **Test entity resolution** algorithms on the data
3. **Validate results** using ground truth
4. **Analyze** which error types are most challenging

For detailed documentation, see `IMPLEMENTATION_SUMMARY.md` and `README.md`.
