"""Unit tests for shared/ground_truth.py — true pair generation from ground truth."""

import pandas as pd
import pytest

from shared.ground_truth import (
    add_record_ids_to_ground_truth,
    generate_true_pairs_from_ground_truth,
)


@pytest.mark.unit
class TestGenerateTruePairs:
    def test_two_facilities(self):
        """Patient at 2 facilities -> 1 pair."""
        gt = pd.DataFrame(
            {
                "true_patient_id": ["p1", "p1"],
                "record_id": ["fac1_p1", "fac2_p1"],
            }
        )
        pairs = generate_true_pairs_from_ground_truth(gt)
        assert len(pairs) == 1
        assert ("fac1_p1", "fac2_p1") in pairs

    def test_three_facilities(self):
        """Patient at 3 facilities -> C(3,2) = 3 pairs."""
        gt = pd.DataFrame(
            {
                "true_patient_id": ["p1", "p1", "p1"],
                "record_id": ["fac1_p1", "fac2_p1", "fac3_p1"],
            }
        )
        pairs = generate_true_pairs_from_ground_truth(gt)
        assert len(pairs) == 3

    def test_singletons_ignored(self):
        """Patient at only 1 facility -> 0 pairs."""
        gt = pd.DataFrame(
            {
                "true_patient_id": ["p1"],
                "record_id": ["fac1_p1"],
            }
        )
        pairs = generate_true_pairs_from_ground_truth(gt)
        assert len(pairs) == 0

    def test_multiple_patients(self):
        """2 patients, each at 2 facilities -> 2 pairs."""
        gt = pd.DataFrame(
            {
                "true_patient_id": ["p1", "p1", "p2", "p2"],
                "record_id": ["fac1_p1", "fac2_p1", "fac1_p2", "fac2_p2"],
            }
        )
        pairs = generate_true_pairs_from_ground_truth(gt)
        assert len(pairs) == 2
        assert ("fac1_p1", "fac2_p1") in pairs
        assert ("fac1_p2", "fac2_p2") in pairs

    def test_uses_original_patient_uuid_fallback(self):
        """Falls back to original_patient_uuid column when true_patient_id absent."""
        gt = pd.DataFrame(
            {
                "original_patient_uuid": ["p1", "p1"],
                "record_id": ["fac1_p1", "fac2_p1"],
            }
        )
        pairs = generate_true_pairs_from_ground_truth(gt)
        assert len(pairs) == 1


@pytest.mark.unit
class TestAddRecordIds:
    def test_merge_adds_record_id(self):
        """record_id column is added to ground truth via join on facility_id + id."""
        gt = pd.DataFrame(
            {
                "facility_id": ["facility_001", "facility_002"],
                "original_patient_uuid": ["uuid-1", "uuid-1"],
            }
        )
        patients = pd.DataFrame(
            {
                "facility_id": ["facility_001", "facility_002"],
                "id": ["uuid-1", "uuid-1"],
                "record_id": ["facility_001_uuid-1", "facility_002_uuid-1"],
            }
        )
        result = add_record_ids_to_ground_truth(gt, patients)
        assert "record_id" in result.columns
        assert result["record_id"].tolist() == [
            "facility_001_uuid-1",
            "facility_002_uuid-1",
        ]
        assert "true_patient_id" in result.columns
