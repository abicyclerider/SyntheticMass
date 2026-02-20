# Synthea Runner

Docker-based setup for generating synthetic patient data using [Synthea](https://github.com/synthetichealth/synthea), managed by DVC.

## Overview

This runner uses the [synthea-docker](https://github.com/mrreband/synthea-docker) submodule to run Synthea in a containerized environment, generating realistic synthetic medical records for testing and development.

## Prerequisites

- **Docker**: For running Synthea
- **Git**: For cloning repository and initializing submodules
- **8GB+ RAM**: Required for Synthea generation (Java heap set to 8GB)

## Quick Start

### 1. Initialize Submodule (if not already done)

```bash
# From repository root
git submodule update --init --recursive
```

### 2. Generate Synthetic Patients

Generation is managed by DVC stages (`generate` and `generate_training`):

```bash
# Run via DVC (recommended)
dvc repro generate

# Output will be created in: ./output/synthea_raw/csv/
```

Generation parameters are configured in [`params.yaml`](../params.yaml) and run via DVC:
- **2,500 patients** for inference (seed 67890)
- **30,000 patients** for training (seed 12345)
- **8GB RAM** Java heap allocation
- **CSV format only** (FHIR disabled)
- Massachusetts population

## Generated Output

Synthea produces CSV files in `./output/synthea_raw/csv/`:

### Core Files
- `patients.csv` - Patient demographics (name, DOB, SSN, address, etc.)
- `encounters.csv` - Medical visits and encounters
- `conditions.csv` - Diagnoses and medical conditions
- `medications.csv` - Prescriptions and medication orders
- `observations.csv` - Lab results and vital signs
- `procedures.csv` - Medical procedures and surgeries

### Additional Files
- `allergies.csv` - Patient allergies
- `careplans.csv` - Care plans and treatment protocols
- `devices.csv` - Medical devices
- `imaging_studies.csv` - Radiology and imaging data
- `immunizations.csv` - Vaccination records
- `organizations.csv` - Healthcare organizations
- `payers.csv` - Insurance payers
- `providers.csv` - Healthcare providers

## Configuration

### Modify Generation Parameters

Edit `params.yaml` in the project root to change generation parameters:
- `generate.population`: Patient count for inference (default: 2500)
- `generate.seed`: Random seed for inference (default: 67890)
- `generate_training.population`: Patient count for training (default: 30000)
- `generate_training.seed`: Random seed for training (default: 12345)

### Synthea Properties

The `config/synthea.properties` file customizes:
- CSV export settings
- Socioeconomic distributions
- Demographics and population characteristics
- Clinical modules and conditions

See [Synthea Configuration Guide](https://github.com/synthetichealth/synthea/wiki/Common-Configuration) for all options.

## Directory Structure

```
synthea_runner/
├── config/
│   └── synthea.properties         # Synthea configuration
├── synthea-docker/                # Synthea Docker submodule (Dockerfile + build)
└── output/                        # Generated data (gitignored)
    ├── synthea_raw/csv/           # Synthea CSV output
    └── synthea_parquet/           # Parquet-converted output (from csv_to_parquet stage)
```

## Documentation

- [Synthea Wiki](https://github.com/synthetichealth/synthea/wiki) - Complete documentation
- [CSV Data Dictionary](https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary) - Schema reference
- [Common Configuration](https://github.com/synthetichealth/synthea/wiki/Common-Configuration) - Configuration guide
- [synthea-docker Repo](https://github.com/mrreband/synthea-docker) - Docker wrapper documentation

## Troubleshooting

### Submodule not initialized
```bash
# From repository root
git submodule update --init --recursive
```

### Out of memory errors
The DVC stage sets `JAVA_OPTS=-Xmx8g` by default. For larger populations, increase the heap size in the `generate` stage command in `dvc.yaml`.

### Permission errors on output directory
```bash
chmod -R 755 output/
```

---

**Built with**: [Synthea](https://github.com/synthetichealth/synthea) | [synthea-docker](https://github.com/mrreband/synthea-docker)
