"""Seam 3: ER pipeline end-to-end through Splink."""

import pandas as pd
import pytest

from entity_resolution.core.golden_record import build_match_clusters
from entity_resolution.core.splink_linker import (
    classify_predictions,
    create_linker,
    predict_matches,
    train_model,
)
from entity_resolution.resolve import prepare_for_splink
from shared.data_loader import create_record_id, load_facility_patients
from shared.evaluation import calculate_confusion_matrix, calculate_metrics
from shared.ground_truth import (
    add_record_ids_to_ground_truth,
    generate_true_pairs_from_ground_truth,
    load_ground_truth,
)

SPLINK_CONFIG = {
    "splink": {
        "predict_threshold": 0.01,
        "auto_match_probability": 0.90,
        "auto_reject_probability": 0.05,
    },
}


def _load_and_prepare(augmented_run_dir):
    """Load patients, add record_id, and prepare for Splink."""
    patients = create_record_id(load_facility_patients(str(augmented_run_dir)))
    splink_df = prepare_for_splink(patients)
    return patients, splink_df


def _get_true_pairs(augmented_run_dir, patients):
    """Load ground truth and generate true pairs."""
    gt = load_ground_truth(str(augmented_run_dir))
    gt = add_record_ids_to_ground_truth(gt, patients)
    return generate_true_pairs_from_ground_truth(gt)


@pytest.mark.integration
class TestERPipeline:
    def test_splink_finds_known_matches(self, augmented_run_dir):
        patients, splink_df = _load_and_prepare(augmented_run_dir)
        true_pairs = _get_true_pairs(augmented_run_dir, patients)

        linker, _ = create_linker(splink_df, SPLINK_CONFIG)
        train_model(linker)
        predictions = predict_matches(linker, SPLINK_CONFIG)

        # Build pair → probability lookup
        pred_pairs = {}
        for _, row in predictions.iterrows():
            pair = tuple(sorted([row["record_id_l"], row["record_id_r"]]))
            pred_pairs[pair] = row["match_probability"]

        for pair in true_pairs:
            assert pair in pred_pairs, f"True pair {pair} not found in predictions"
            assert pred_pairs[pair] > 0.90, (
                f"Match probability too low for {pair}: {pred_pairs[pair]:.4f}"
            )

    def test_classify_then_golden_records(self, augmented_run_dir):
        patients, splink_df = _load_and_prepare(augmented_run_dir)

        linker, _ = create_linker(splink_df, SPLINK_CONFIG)
        train_model(linker)
        predictions = predict_matches(linker, SPLINK_CONFIG)
        auto_matches, _, _ = classify_predictions(predictions, SPLINK_CONFIG)

        # Build boolean Series for build_match_clusters
        match_index = pd.MultiIndex.from_frame(
            auto_matches[["record_id_1", "record_id_2"]]
        )
        match_series = pd.Series(True, index=match_index)
        clusters = build_match_clusters(match_series)

        # 2 matched clusters + 1 singleton = 3 unique patients
        matched_ids = set()
        for cluster in clusters:
            matched_ids.update(cluster)

        singleton_ids = set(patients["record_id"]) - matched_ids
        total_clusters = len(clusters) + len(singleton_ids)
        assert total_clusters == 3

    def test_full_pipeline_f1(self, augmented_run_dir):
        patients, splink_df = _load_and_prepare(augmented_run_dir)
        true_pairs = _get_true_pairs(augmented_run_dir, patients)

        linker, _ = create_linker(splink_df, SPLINK_CONFIG)
        train_model(linker)
        predictions = predict_matches(linker, SPLINK_CONFIG)
        auto_matches, _, _ = classify_predictions(predictions, SPLINK_CONFIG)

        predicted_pairs = set()
        for _, row in auto_matches.iterrows():
            pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
            predicted_pairs.add(pair)

        tp, fp, fn = calculate_confusion_matrix(predicted_pairs, true_pairs)
        metrics = calculate_metrics(tp, fp, fn)
        assert metrics["f1_score"] == 1.0
