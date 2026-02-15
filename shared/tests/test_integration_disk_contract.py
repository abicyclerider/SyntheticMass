"""Seam 1: Disk contract between augmentation output and shared loaders."""

import pytest

from shared.data_loader import load_facility_patients
from shared.ground_truth import load_ground_truth
from shared.medical_records import load_medical_records


@pytest.mark.integration
class TestDiskContract:
    def test_load_facility_patients_reads_all_facilities(self, augmented_run_dir):
        df = load_facility_patients(str(augmented_run_dir))
        assert len(df) == 5
        assert set(df["facility_id"]) == {
            "facility_001",
            "facility_002",
            "facility_003",
        }

    def test_load_facility_patients_standardizes_columns(self, augmented_run_dir):
        df = load_facility_patients(str(augmented_run_dir))
        # Title-cased names
        assert df["first_name"].str[0].eq(df["first_name"].str[0].str.upper()).all()
        # SSN dashes removed
        assert not df["ssn"].str.contains("-").any()
        # ZIP zero-padded to 5 digits
        assert (df["zip"].str.len() == 5).all()
        # State uppercased
        assert df["state"].eq(df["state"].str.upper()).all()

    def test_load_ground_truth_normalizes_facility_id(self, augmented_run_dir):
        gt = load_ground_truth(str(augmented_run_dir))
        for fid in gt["facility_id"]:
            assert fid.startswith("facility_")
            assert len(fid) == len("facility_001")

    def test_load_medical_records_reads_clinical_tables(self, augmented_run_dir):
        records = load_medical_records(str(augmented_run_dir))
        for table in [
            "conditions",
            "encounters",
            "medications",
            "observations",
            "allergies",
            "procedures",
        ]:
            assert table in records, f"Missing clinical table: {table}"
            df = records[table]
            assert "facility_id" in df.columns
            assert "PATIENT" in df.columns

    def test_facility_ids_consistent_across_loaders(self, augmented_run_dir):
        patients = load_facility_patients(str(augmented_run_dir))
        gt = load_ground_truth(str(augmented_run_dir))
        records = load_medical_records(str(augmented_run_dir))

        patient_fids = set(patients["facility_id"])
        gt_fids = set(gt["facility_id"])

        assert patient_fids == gt_fids

        for table_name, df in records.items():
            record_fids = set(df["facility_id"])
            assert record_fids <= patient_fids, (
                f"{table_name} has unexpected facility_ids"
            )
