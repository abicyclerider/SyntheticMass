"""Unit tests for shared/data_loader.py — column standardization and record ID creation."""

import pandas as pd
import pytest

from shared.data_loader import create_record_id, standardize_columns


@pytest.mark.unit
class TestStandardizeColumns:
    def test_renames_synthea_columns(self, sample_patients_raw):
        """Synthea columns (Id, FIRST, LAST, etc.) are mapped to lowercase names."""
        result = standardize_columns(sample_patients_raw)
        for expected_col in [
            "id",
            "first_name",
            "last_name",
            "birthdate",
            "ssn",
            "address",
            "city",
            "state",
            "zip",
            "gender",
        ]:
            assert expected_col in result.columns

    def test_title_case_names(self, sample_patients_raw):
        """Names are converted to title case: 'john' -> 'John', 'SMITH' -> 'Smith'."""
        result = standardize_columns(sample_patients_raw)
        assert result["first_name"].tolist() == [
            "John",
            "Jane",
            "Bob",
            "Alice",
            "Charlie",
        ]
        assert result["last_name"].tolist() == [
            "Smith",
            "Doe",
            "Jones",
            "Williams",
            "Brown",
        ]

    def test_ssn_dashes_removed(self, sample_patients_raw):
        """SSN dashes are stripped: '123-45-6789' -> '123456789'."""
        result = standardize_columns(sample_patients_raw)
        assert result["ssn"].iloc[0] == "123456789"
        assert "-" not in result["ssn"].iloc[0]

    def test_zip_zero_padded(self, sample_patients_raw):
        """Short ZIP codes are zero-padded to 5 digits: '1101' -> '01101'."""
        result = standardize_columns(sample_patients_raw)
        assert result["zip"].iloc[2] == "01101"
        assert result["zip"].iloc[4] == "01852"
        assert all(result["zip"].str.len() == 5)

    def test_birthdate_parsed(self, sample_patients_raw):
        """Birthdate strings are parsed to datetime and birth_year is added."""
        result = standardize_columns(sample_patients_raw)
        assert pd.api.types.is_datetime64_any_dtype(result["birthdate"])
        assert result["birth_year"].iloc[0] == 1980
        assert result["birth_year"].iloc[1] == 1990

    def test_state_uppercased(self, sample_patients_raw):
        """State codes are uppercased: 'ma' -> 'MA'."""
        result = standardize_columns(sample_patients_raw)
        assert all(result["state"] == "MA")

    def test_does_not_mutate_input(self, sample_patients_raw):
        """Standardization returns a copy, not a mutation of the input."""
        original_first = sample_patients_raw["FIRST"].tolist()
        standardize_columns(sample_patients_raw)
        assert sample_patients_raw["FIRST"].tolist() == original_first


@pytest.mark.unit
class TestCreateRecordId:
    def test_creates_record_id(self):
        """record_id = facility_id + '_' + id."""
        df = pd.DataFrame(
            {"facility_id": ["fac_001", "fac_002"], "id": ["uuid-1", "uuid-2"]}
        )
        result = create_record_id(df)
        assert result["record_id"].tolist() == ["fac_001_uuid-1", "fac_002_uuid-2"]

    def test_does_not_mutate_input(self):
        """create_record_id returns a copy."""
        df = pd.DataFrame({"facility_id": ["fac_001"], "id": ["uuid-1"]})
        result = create_record_id(df)
        assert "record_id" not in df.columns
        assert "record_id" in result.columns
