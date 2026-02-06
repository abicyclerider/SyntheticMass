"""
Data loading and preparation for entity resolution.

Loads patient records from multiple facility CSVs and ground truth for validation.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_facility_patients(run_dir: str) -> pd.DataFrame:
    """
    Load patient records from all facility CSVs into a single DataFrame.

    Args:
        run_dir: Path to augmentation run directory (e.g., output/augmented/run_20260202_122731)

    Returns:
        DataFrame with all patient records, including facility_id column
    """
    run_path = Path(run_dir)
    facilities_dir = run_path / "facilities"

    if not facilities_dir.exists():
        raise FileNotFoundError(f"Facilities directory not found: {facilities_dir}")

    all_patients = []
    facility_dirs = sorted([d for d in facilities_dir.iterdir() if d.is_dir()])

    logger.info(f"Loading patients from {len(facility_dirs)} facilities...")

    for facility_dir in facility_dirs:
        facility_id = facility_dir.name
        patients_file = facility_dir / "patients.csv"

        if not patients_file.exists():
            logger.warning(f"No patients.csv found in {facility_id}")
            continue

        df = pd.read_csv(patients_file)
        df['facility_id'] = facility_id
        all_patients.append(df)
        logger.debug(f"Loaded {len(df)} patients from {facility_id}")

    if not all_patients:
        raise ValueError("No patient records found")

    combined_df = pd.concat(all_patients, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total patient records from {len(facility_dirs)} facilities")

    # Standardize and clean
    combined_df = prepare_for_matching(combined_df)

    return combined_df


def load_ground_truth(run_dir: str) -> pd.DataFrame:
    """
    Load ground truth mapping of patient records to true patient IDs.

    Args:
        run_dir: Path to augmentation run directory

    Returns:
        DataFrame with columns: facility_id, patient_id, true_patient_id
    """
    run_path = Path(run_dir)
    ground_truth_file = run_path / "metadata" / "ground_truth.csv"

    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    df = pd.read_csv(ground_truth_file)
    logger.info(f"Loaded ground truth with {len(df)} records")

    return df


def prepare_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare patient data for matching by standardizing fields and cleaning.

    Args:
        df: Raw patient DataFrame

    Returns:
        Cleaned DataFrame ready for entity resolution
    """
    df = df.copy()

    # Strip whitespace from string columns
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Standardize field names to lowercase for consistency
    # Synthea uses: Id, BIRTHDATE, DEATHDATE, SSN, DRIVERS, PASSPORT, PREFIX, FIRST, LAST,
    # MAIDEN, MARITAL, RACE, ETHNICITY, GENDER, BIRTHPLACE, ADDRESS, CITY, STATE, COUNTY, ZIP, LAT, LON

    # Create standardized field names for matching
    field_mapping = {
        'Id': 'id',
        'BIRTHDATE': 'birthdate',
        'SSN': 'ssn',
        'FIRST': 'first_name',
        'LAST': 'last_name',
        'MAIDEN': 'maiden_name',
        'ADDRESS': 'address',
        'CITY': 'city',
        'STATE': 'state',
        'ZIP': 'zip',
        'GENDER': 'gender'
    }

    for old_name, new_name in field_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Normalize case for names (title case)
    for col in ['first_name', 'last_name', 'maiden_name']:
        if col in df.columns:
            df[col] = df[col].str.title()

    # Normalize SSN format (remove dashes if present)
    if 'ssn' in df.columns:
        df['ssn'] = df[col].str.replace('-', '', regex=False)

    # Convert birthdate to datetime
    if 'birthdate' in df.columns:
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')

    # Normalize ZIP codes (ensure 5 digits)
    if 'zip' in df.columns:
        df['zip'] = df['zip'].astype(str).str.zfill(5)

    # Uppercase state codes
    if 'state' in df.columns:
        df['state'] = df['state'].str.upper()

    logger.debug(f"Prepared {len(df)} records for matching")

    return df


def create_record_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unique record identifier for each patient-facility combination.

    Args:
        df: Patient DataFrame with facility_id and id columns

    Returns:
        DataFrame with added record_id column
    """
    df = df.copy()
    df['record_id'] = df['facility_id'] + '_' + df['id'].astype(str)
    return df


def get_run_directory(base_dir: str, run_id: str) -> Path:
    """
    Get the full path to a run directory.

    Args:
        base_dir: Base directory of the project
        run_id: Run identifier (e.g., run_20260202_122731)

    Returns:
        Path to run directory
    """
    run_dir = Path(base_dir) / "output" / "augmented" / run_id

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return run_dir


def load_data_for_matching(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all necessary data for entity resolution from configuration.

    Args:
        config: Configuration dictionary with base_dir and run_id

    Returns:
        Tuple of (patient_records_df, ground_truth_df)
    """
    run_dir = get_run_directory(config['base_dir'], config['run_id'])

    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)

    ground_truth_df = load_ground_truth(str(run_dir))

    logger.info(f"Loaded {len(patients_df)} patient records and {len(ground_truth_df)} ground truth entries")

    return patients_df, ground_truth_df
