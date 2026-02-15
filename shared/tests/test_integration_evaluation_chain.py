"""Seam 2: Evaluation chain from ground truth through to metrics."""

import pytest

from shared.data_loader import create_record_id, load_facility_patients
from shared.evaluation import calculate_confusion_matrix, calculate_metrics
from shared.ground_truth import (
    add_record_ids_to_ground_truth,
    generate_true_pairs_from_ground_truth,
    load_ground_truth,
)


@pytest.mark.integration
class TestEvaluationChain:
    def test_true_pairs_count_matches_expected(self, augmented_run_dir):
        patients = create_record_id(load_facility_patients(str(augmented_run_dir)))
        gt = load_ground_truth(str(augmented_run_dir))
        gt = add_record_ids_to_ground_truth(gt, patients)
        true_pairs = generate_true_pairs_from_ground_truth(gt)
        # uuid-A across 2 facilities + uuid-B across 2 facilities → 2 pairs
        assert len(true_pairs) == 2

    def test_record_ids_join_correctly(self, augmented_run_dir):
        patients = create_record_id(load_facility_patients(str(augmented_run_dir)))
        gt = load_ground_truth(str(augmented_run_dir))
        gt = add_record_ids_to_ground_truth(gt, patients)
        assert "record_id" in gt.columns
        assert gt["record_id"].notna().all()

    def test_perfect_prediction_yields_perfect_f1(self, augmented_run_dir):
        patients = create_record_id(load_facility_patients(str(augmented_run_dir)))
        gt = load_ground_truth(str(augmented_run_dir))
        gt = add_record_ids_to_ground_truth(gt, patients)
        true_pairs = generate_true_pairs_from_ground_truth(gt)

        tp, fp, fn = calculate_confusion_matrix(true_pairs, true_pairs)
        metrics = calculate_metrics(tp, fp, fn)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_partial_prediction_metrics(self, augmented_run_dir):
        patients = create_record_id(load_facility_patients(str(augmented_run_dir)))
        gt = load_ground_truth(str(augmented_run_dir))
        gt = add_record_ids_to_ground_truth(gt, patients)
        true_pairs = generate_true_pairs_from_ground_truth(gt)

        # Predict only 1 of 2 true pairs → P=1.0, R=0.5
        predicted = {next(iter(true_pairs))}
        tp, fp, fn = calculate_confusion_matrix(predicted, true_pairs)
        metrics = calculate_metrics(tp, fp, fn)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5
