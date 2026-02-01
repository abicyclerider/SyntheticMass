# SyntheticMass

Medical data generation and augmentation pipeline for entity resolution training.

## Overview

This repository contains tools for generating and augmenting synthetic medical datasets. The goal is to create realistic patient data with controlled variations and errors for training and evaluating entity resolution algorithms.

## Project Structure

```
SyntheticMass/
├── synthea-runner/          # Synthea patient data generation
│   ├── docker-compose.yml   # Docker Compose configuration
│   ├── config/              # Synthea configuration
│   ├── synthea-docker/      # Synthea Docker submodule
│   ├── output/              # Generated synthetic patients (gitignored)
│   └── README.md            # Synthea runner documentation
│
├── (future) augmentation/   # Data augmentation pipeline
│   └── ...                  # Error injection, duplicates, ground truth
│
├── LICENSE                  # Project license
└── README.md                # This file
```

## Components

### 1. Synthea Runner (`synthea-runner/`)

Generates base synthetic patient records using [Synthea](https://github.com/synthetichealth/synthea).

**Features:**
- Generates realistic medical records (demographics, encounters, conditions, medications, etc.)
- Configurable population size, demographics, and clinical modules
- CSV export format
- Reproducible output with fixed random seeds

**Quick Start:**
```bash
# Initialize submodules
git submodule update --init --recursive

# Generate patients
cd synthea-runner
docker compose up
```

See [`synthea-runner/README.md`](synthea-runner/README.md) for full documentation.

### 2. Data Augmentation (Coming Soon)

Future component for augmenting Synthea data with:
- Controlled errors and variations (typos, formatting differences, missing fields)
- Duplicate record generation
- Ground truth tracking for entity resolution training
- Configurable error rates based on healthcare research

## Getting Started

### Prerequisites

- **Docker**: For running Synthea
- **Git**: For cloning repository and submodules
- **4GB+ RAM**: Required for Synthea generation

### Initial Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SyntheticMass.git
cd SyntheticMass

# Initialize submodules
git submodule update --init --recursive

# Generate base patient data
cd synthea-runner
docker compose up
```

## Workflow

1. **Generate Base Data** - Use synthea-runner to create synthetic patients
2. **Augment Data** (future) - Add controlled errors and duplicates
3. **Train Models** (external) - Use augmented data for entity resolution

## License

See `LICENSE` file for details.

## Documentation

- [Synthea Runner Documentation](synthea-runner/README.md)
- [Synthea Official Wiki](https://github.com/synthetichealth/synthea/wiki)

---

**Current Status**: Base patient generation operational via synthea-runner
