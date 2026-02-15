"""Shared test fixtures for entity resolution tests."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_patients_standardized():
    """Sample patient DataFrame in standardized format for entity resolution tests."""
    return pd.DataFrame(
        {
            "record_id": [
                "fac1_p1",
                "fac2_p1",
                "fac1_p2",
                "fac2_p2",
                "fac1_p3",
            ],
            "facility_id": ["fac1", "fac2", "fac1", "fac2", "fac1"],
            "id": ["p1", "p1", "p2", "p2", "p3"],
            "first_name": ["John", "John", "Jane", "Jane", "Bob"],
            "last_name": ["Smith", "Smith", "Doe", "Doe", "Jones"],
            "address": [
                "123 Main St",
                "123 Main St",
                "456 Oak Ave",
                "456 Oak Ave",
                "789 Pine Rd",
            ],
            "city": ["Boston", "Boston", "Boston", "Boston", "Springfield"],
            "state": ["MA", "MA", "MA", "MA", "MA"],
            "zip": ["02115", "02115", "02116", "02116", "01101"],
            "ssn": [
                "111-22-3333",
                "111-22-3333",
                "444-55-6666",
                "444-55-6666",
                "777-88-9999",
            ],
            "birthdate": pd.to_datetime(
                [
                    "1980-01-15",
                    "1980-01-15",
                    "1990-06-20",
                    "1990-06-20",
                    "1975-03-10",
                ]
            ),
            "birth_year": [1980, 1980, 1990, 1990, 1975],
            "gender": ["M", "M", "F", "F", "M"],
            "maiden_name": ["", "", "", "", ""],
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
