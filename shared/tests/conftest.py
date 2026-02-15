"""Shared test fixtures for shared module tests."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_patients_raw():
    """Sample patient DataFrame with raw Synthea column names (pre-standardization)."""
    return pd.DataFrame(
        {
            "Id": ["uuid-1", "uuid-2", "uuid-3", "uuid-4", "uuid-5"],
            "FIRST": ["john", "JANE", "bob", "Alice", "Charlie"],
            "LAST": ["SMITH", "doe", "JONES", "Williams", "Brown"],
            "MAIDEN": ["", "", "", "Taylor", ""],
            "BIRTHDATE": [
                "1980-01-15",
                "1990-06-20",
                "1975-03-10",
                "1985-11-25",
                "2000-07-04",
            ],
            "SSN": [
                "123-45-6789",
                "234-56-7890",
                "345-67-8901",
                "456-78-9012",
                "567-89-0123",
            ],
            "ADDRESS": [
                "123 Main St",
                "456 Oak Ave",
                "789 Pine Rd",
                "321 Elm St",
                "654 Maple Dr",
            ],
            "CITY": ["Boston", "Cambridge", "Springfield", "Worcester", "Lowell"],
            "STATE": ["ma", "MA", "Ma", "ma", "MA"],
            "ZIP": ["02115", "02139", "1101", "01609", "1852"],
            "GENDER": ["M", "F", "M", "F", "M"],
            "facility_id": [
                "facility_001",
                "facility_001",
                "facility_002",
                "facility_002",
                "facility_003",
            ],
        }
    )


@pytest.fixture
def sample_patients_standardized():
    """Sample patient DataFrame in standardized format (post-standardization)."""
    return pd.DataFrame(
        {
            "id": ["uuid-1", "uuid-2", "uuid-3", "uuid-4", "uuid-5"],
            "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
            "last_name": ["Smith", "Doe", "Jones", "Williams", "Brown"],
            "birthdate": pd.to_datetime(
                [
                    "1980-01-15",
                    "1990-06-20",
                    "1975-03-10",
                    "1985-11-25",
                    "2000-07-04",
                ]
            ),
            "birth_year": [1980, 1990, 1975, 1985, 2000],
            "ssn": [
                "123456789",
                "234567890",
                "345678901",
                "456789012",
                "567890123",
            ],
            "address": [
                "123 Main St",
                "456 Oak Ave",
                "789 Pine Rd",
                "321 Elm St",
                "654 Maple Dr",
            ],
            "city": ["Boston", "Cambridge", "Springfield", "Worcester", "Lowell"],
            "state": ["MA", "MA", "MA", "MA", "MA"],
            "zip": ["02115", "02139", "01101", "01609", "01852"],
            "gender": ["M", "F", "M", "F", "M"],
            "facility_id": [
                "facility_001",
                "facility_001",
                "facility_002",
                "facility_002",
                "facility_003",
            ],
            "record_id": [
                "facility_001_uuid-1",
                "facility_001_uuid-2",
                "facility_002_uuid-3",
                "facility_002_uuid-4",
                "facility_003_uuid-5",
            ],
        }
    )


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth DataFrame mapping facility records to true patient IDs."""
    return pd.DataFrame(
        {
            "facility_id": [
                "facility_001",
                "facility_002",
                "facility_001",
                "facility_002",
                "facility_003",
            ],
            "original_patient_uuid": [
                "uuid-1",
                "uuid-1",
                "uuid-2",
                "uuid-2",
                "uuid-3",
            ],
            "true_patient_id": [
                "true-1",
                "true-1",
                "true-2",
                "true-2",
                "true-3",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Integration test fixture: realistic augmentation output directory
# ---------------------------------------------------------------------------

_PATIENTS = {
    "uuid-A": {
        "FIRST": "John",
        "LAST": "Smith",
        "MAIDEN": "",
        "BIRTHDATE": "1980-01-15",
        "SSN": "123-45-6789",
        "ADDRESS": "123 Main St",
        "CITY": "Boston",
        "STATE": "MA",
        "ZIP": "02115",
        "GENDER": "M",
    },
    "uuid-B": {
        "FIRST": "Jane",
        "LAST": "Doe",
        "MAIDEN": "Taylor",
        "BIRTHDATE": "1990-06-20",
        "SSN": "234-56-7890",
        "ADDRESS": "456 Oak Ave",
        "CITY": "Cambridge",
        "STATE": "MA",
        "ZIP": "02139",
        "GENDER": "F",
    },
    "uuid-C": {
        "FIRST": "Bob",
        "LAST": "Jones",
        "MAIDEN": "",
        "BIRTHDATE": "1975-03-10",
        "SSN": "345-67-8901",
        "ADDRESS": "789 Pine Rd",
        "CITY": "Springfield",
        "STATE": "MA",
        "ZIP": "01101",
        "GENDER": "M",
    },
}

_FACILITY_PATIENTS = {
    "facility_001": ["uuid-A", "uuid-B"],
    "facility_002": ["uuid-A", "uuid-C"],
    "facility_003": ["uuid-B"],
}


def _write_clinical_records(fac_dir: Path, patient_ids: list[str]) -> None:
    """Write minimal clinical parquet files for integration tests."""
    enc, cond, med, obs, allg, proc = [], [], [], [], [], []

    for pid in patient_ids:
        enc.extend(
            [
                {
                    "Id": f"enc-{pid}-1",
                    "START": "2020-03-15T10:00:00Z",
                    "STOP": "2020-03-15T11:00:00Z",
                    "PATIENT": pid,
                    "ENCOUNTERCLASS": "ambulatory",
                    "DESCRIPTION": "General examination",
                },
                {
                    "Id": f"enc-{pid}-2",
                    "START": "2021-06-20T09:00:00Z",
                    "STOP": "2021-06-20T10:00:00Z",
                    "PATIENT": pid,
                    "ENCOUNTERCLASS": "ambulatory",
                    "DESCRIPTION": "Follow-up visit",
                },
            ]
        )
        cond.extend(
            [
                {
                    "START": "2020-03-15",
                    "STOP": "",
                    "PATIENT": pid,
                    "ENCOUNTER": f"enc-{pid}-1",
                    "DESCRIPTION": "Essential hypertension",
                    "CODE": "59621000",
                },
                {
                    "START": "2021-06-20",
                    "STOP": "2021-12-01",
                    "PATIENT": pid,
                    "ENCOUNTER": f"enc-{pid}-2",
                    "DESCRIPTION": "Acute bronchitis",
                    "CODE": "10509002",
                },
            ]
        )
        med.append(
            {
                "START": "2020-03-15",
                "STOP": "",
                "PATIENT": pid,
                "ENCOUNTER": f"enc-{pid}-1",
                "DESCRIPTION": "Lisinopril 10 MG Oral Tablet",
                "CODE": "314076",
            }
        )
        obs.extend(
            [
                {
                    "DATE": "2020-03-15",
                    "PATIENT": pid,
                    "ENCOUNTER": f"enc-{pid}-1",
                    "DESCRIPTION": "Body Height",
                    "VALUE": "170",
                    "UNITS": "cm",
                    "CODE": "8302-2",
                },
                {
                    "DATE": "2020-03-15",
                    "PATIENT": pid,
                    "ENCOUNTER": f"enc-{pid}-1",
                    "DESCRIPTION": "Body Weight",
                    "VALUE": "75",
                    "UNITS": "kg",
                    "CODE": "29463-7",
                },
                {
                    "DATE": "2021-06-20",
                    "PATIENT": pid,
                    "ENCOUNTER": f"enc-{pid}-2",
                    "DESCRIPTION": "Body Weight",
                    "VALUE": "77",
                    "UNITS": "kg",
                    "CODE": "29463-7",
                },
            ]
        )
        allg.append(
            {
                "START": "2020-03-15",
                "STOP": "",
                "PATIENT": pid,
                "ENCOUNTER": f"enc-{pid}-1",
                "DESCRIPTION": "Penicillin V",
                "CODE": "7980",
            }
        )
        proc.append(
            {
                "START": "2020-03-15",
                "STOP": "2020-03-15",
                "PATIENT": pid,
                "ENCOUNTER": f"enc-{pid}-1",
                "DESCRIPTION": "Measurement of respiratory function",
                "CODE": "23426006",
            }
        )

    for name, rows in [
        ("encounters", enc),
        ("conditions", cond),
        ("medications", med),
        ("observations", obs),
        ("allergies", allg),
        ("procedures", proc),
    ]:
        pd.DataFrame(rows).to_parquet(fac_dir / f"{name}.parquet", index=False)


@pytest.fixture
def augmented_run_dir(tmp_path):
    """Build a realistic augmentation output directory.

    Layout (5 records, 3 true patients, 2 match pairs):
      - uuid-A at facility_001 + facility_002
      - uuid-B at facility_001 + facility_003
      - uuid-C at facility_002 only (singleton)
    """
    for fac_name, pat_ids in _FACILITY_PATIENTS.items():
        fac_dir = tmp_path / "facilities" / fac_name
        fac_dir.mkdir(parents=True)
        rows = [{"Id": pid, **_PATIENTS[pid]} for pid in pat_ids]
        pd.DataFrame(rows).to_parquet(fac_dir / "patients.parquet", index=False)
        _write_clinical_records(fac_dir, pat_ids)

    # Ground truth with integer facility_id (as augmentation writes it)
    gt_rows = []
    for fac_name, pat_ids in _FACILITY_PATIENTS.items():
        fac_int = int(fac_name.split("_")[1])
        for pid in pat_ids:
            gt_rows.append({"facility_id": fac_int, "original_patient_uuid": pid})

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(gt_rows).to_parquet(metadata_dir / "ground_truth.parquet", index=False)

    return tmp_path
