"""Unit tests for shared/evaluation.py — pure metric calculations."""

import pytest

from shared.evaluation import calculate_confusion_matrix, calculate_metrics


@pytest.mark.unit
class TestCalculateConfusionMatrix:
    def test_perfect_prediction(self):
        """All predicted == all true -> TP only, no FP/FN."""
        true = {("a", "b"), ("c", "d")}
        predicted = {("a", "b"), ("c", "d")}
        tp, fp, fn = calculate_confusion_matrix(predicted, true)
        assert (tp, fp, fn) == (2, 0, 0)

    def test_no_predictions(self):
        """Empty predicted -> 0 TP, 0 FP, all FN."""
        true = {("a", "b"), ("c", "d")}
        predicted = set()
        tp, fp, fn = calculate_confusion_matrix(predicted, true)
        assert (tp, fp, fn) == (0, 0, 2)

    def test_no_true_pairs(self):
        """Empty true -> all predictions are FP."""
        true = set()
        predicted = {("a", "b")}
        tp, fp, fn = calculate_confusion_matrix(predicted, true)
        assert (tp, fp, fn) == (0, 1, 0)

    def test_partial_overlap(self):
        """2 out of 3 predicted are correct, 1 true pair missed."""
        true = {("a", "b"), ("c", "d"), ("e", "f")}
        predicted = {("a", "b"), ("c", "d"), ("g", "h")}
        tp, fp, fn = calculate_confusion_matrix(predicted, true)
        assert (tp, fp, fn) == (2, 1, 1)

    def test_pair_normalization(self):
        """(A,B) and (B,A) are treated as the same pair."""
        true = {("a", "b")}
        predicted = {("b", "a")}
        tp, fp, fn = calculate_confusion_matrix(predicted, true)
        assert (tp, fp, fn) == (1, 0, 0)


@pytest.mark.unit
class TestCalculateMetrics:
    def test_perfect_scores(self):
        metrics = calculate_metrics(tp=5, fp=0, fn=0)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_zero_division_all_zeros(self):
        """tp=fp=fn=0 -> all metrics 0.0 (no division error)."""
        metrics = calculate_metrics(tp=0, fp=0, fn=0)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_partial_overlap_values(self):
        """tp=2, fp=1, fn=1 -> P=2/3, R=2/3, F1=2/3."""
        metrics = calculate_metrics(tp=2, fp=1, fn=1)
        assert metrics["precision"] == pytest.approx(2 / 3)
        assert metrics["recall"] == pytest.approx(2 / 3)
        assert metrics["f1_score"] == pytest.approx(2 / 3)

    def test_no_recall(self):
        """tp=0, fp=0, fn=5 -> P=0, R=0, F1=0."""
        metrics = calculate_metrics(tp=0, fp=0, fn=5)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
