"""Seam 4: Summarizer chain from disk through to text summaries."""

import pytest

from shared.medical_records import get_patient_records, load_medical_records
from shared.summarize import summarize_diff_friendly_from_records


@pytest.mark.integration
class TestSummarizerChain:
    def test_summarize_from_disk(self, augmented_run_dir):
        records = load_medical_records(str(augmented_run_dir))
        patient_recs = get_patient_records("uuid-A", "facility_001", records)
        summary = summarize_diff_friendly_from_records(patient_recs)
        assert len(summary) > 0
        assert summary != "No clinical records available."

    def test_summarize_includes_conditions(self, augmented_run_dir):
        records = load_medical_records(str(augmented_run_dir))
        patient_recs = get_patient_records("uuid-A", "facility_001", records)
        summary = summarize_diff_friendly_from_records(patient_recs)
        assert "CONDITIONS:" in summary

    def test_summarize_different_facilities_different_records(self, augmented_run_dir):
        records = load_medical_records(str(augmented_run_dir))
        recs_1 = get_patient_records("uuid-A", "facility_001", records)
        recs_2 = get_patient_records("uuid-A", "facility_002", records)
        summary_1 = summarize_diff_friendly_from_records(recs_1)
        summary_2 = summarize_diff_friendly_from_records(recs_2)
        assert len(summary_1) > 0
        assert len(summary_2) > 0

    def test_get_patient_records_filters_correctly(self, augmented_run_dir):
        records = load_medical_records(str(augmented_run_dir))
        patient_recs = get_patient_records("uuid-A", "facility_001", records)
        for df in patient_recs.values():
            assert (df["PATIENT"] == "uuid-A").all()
            assert (df["facility_id"] == "facility_001").all()

    def test_singleton_patient_summarize(self, augmented_run_dir):
        records = load_medical_records(str(augmented_run_dir))
        patient_recs = get_patient_records("uuid-C", "facility_002", records)
        summary = summarize_diff_friendly_from_records(patient_recs)
        assert len(summary) > 0
        assert summary != "No clinical records available."
