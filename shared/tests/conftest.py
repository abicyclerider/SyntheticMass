"""Shared test fixtures for shared module tests."""

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
