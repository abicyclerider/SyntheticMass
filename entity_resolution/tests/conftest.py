"""Shared test fixtures for entity resolution tests."""

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
