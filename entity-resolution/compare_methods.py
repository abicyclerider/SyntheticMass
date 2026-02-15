#!/usr/bin/env python3
"""
Compare recordlinkage vs Splink v4 on the same augmented data.

Produces a detailed side-by-side analysis:
  - Blocking: candidate pairs, recall
  - Scoring: distribution of scores, calibration
  - Classification: auto-match/gray-zone/reject counts
  - Accuracy: precision, recall, F1 at various thresholds
  - Timing: wall-clock for each step
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add paths for imports
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_project_root))

from src.blocking import create_candidate_pairs, evaluate_blocking_recall
from src.comparison import add_composite_features, build_comparison_features
from src.data_loader import create_record_id
from src.splink_linker import (
    classify_predictions,
    create_linker,
    evaluate_splink_only,
    predict_matches,
    train_model,
)
from src.blocking import generate_true_pairs_from_ground_truth

from shared.data_loader import load_facility_patients
from shared.ground_truth import add_record_ids_to_ground_truth, load_ground_truth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_prf1(predicted_pairs: set, true_pairs: set) -> dict:
    """Compute precision, recall, F1 from pair sets."""
    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def threshold_sweep(pairs_with_scores: list, true_pairs: set, thresholds: list) -> list:
    """Sweep thresholds and compute metrics at each."""
    results = []
    for thresh in thresholds:
        predicted = {pair for pair, score in pairs_with_scores if score >= thresh}
        m = compute_prf1(predicted, true_pairs)
        m["threshold"] = thresh
        m["n_predicted"] = len(predicted)
        results.append(m)
    return results


def run_recordlinkage(patients_df, patients_indexed, true_pairs, cfg):
    """Run the old recordlinkage pipeline and return metrics."""
    logger.info("=" * 60)
    logger.info("RECORDLINKAGE METHOD")
    logger.info("=" * 60)

    # Blocking
    t0 = time.time()
    candidate_pairs = create_candidate_pairs(patients_indexed, strategy="aggressive_multipass")
    t_blocking = time.time() - t0

    record_id_mapping = patients_df[["record_id", "facility_id", "id"]]
    blocking_metrics = evaluate_blocking_recall(candidate_pairs, ground_truth_df, record_id_mapping)

    # Comparison
    t0 = time.time()
    features = build_comparison_features(candidate_pairs, patients_indexed, cfg)
    features = add_composite_features(features)
    t_comparison = time.time() - t0

    # Classification at multiple thresholds
    thresholds_old = [round(x, 1) for x in np.arange(2.0, 8.1, 0.5)]

    pairs_with_scores = []
    for idx in features.index:
        pair = tuple(sorted([idx[0], idx[1]]))
        pairs_with_scores.append((pair, features.loc[idx, "total_score"]))

    sweep = threshold_sweep(pairs_with_scores, true_pairs, thresholds_old)

    # Find best F1
    best = max(sweep, key=lambda x: x["f1"])

    # Standard thresholds
    auto_match_6 = {pair for pair, s in pairs_with_scores if s >= 6.0}
    auto_match_metrics = compute_prf1(auto_match_6, true_pairs)

    gray_zone_count = sum(1 for _, s in pairs_with_scores if 4.0 <= s < 6.0)

    # Score distribution
    scores = [s for _, s in pairs_with_scores]
    match_scores = [s for pair, s in pairs_with_scores if pair in true_pairs]
    nonmatch_scores = [s for pair, s in pairs_with_scores if pair not in true_pairs]

    return {
        "method": "recordlinkage",
        "blocking_time_s": round(t_blocking, 2),
        "comparison_time_s": round(t_comparison, 2),
        "candidate_pairs": len(candidate_pairs),
        "blocking_recall": blocking_metrics["blocking_recall"],
        "auto_match_count_6.0": len(auto_match_6),
        "auto_match_precision_6.0": round(auto_match_metrics["precision"], 4),
        "auto_match_recall_6.0": round(auto_match_metrics["recall"], 4),
        "auto_match_f1_6.0": round(auto_match_metrics["f1"], 4),
        "gray_zone_count_4.0_6.0": gray_zone_count,
        "best_threshold": best["threshold"],
        "best_f1": round(best["f1"], 4),
        "best_precision": round(best["precision"], 4),
        "best_recall": round(best["recall"], 4),
        "score_mean_matches": round(np.mean(match_scores), 3) if match_scores else None,
        "score_mean_nonmatches": round(np.mean(nonmatch_scores), 3) if nonmatch_scores else None,
        "score_std_matches": round(np.std(match_scores), 3) if match_scores else None,
        "score_std_nonmatches": round(np.std(nonmatch_scores), 3) if nonmatch_scores else None,
        "sweep": sweep,
    }


def run_splink(patients_df, true_pairs, cfg):
    """Run the new Splink pipeline and return metrics."""
    logger.info("=" * 60)
    logger.info("SPLINK METHOD")
    logger.info("=" * 60)

    # Prepare data for Splink (import the canonical prep function from resolve.py)
    sys.path.insert(0, str(_script_dir))
    from resolve import prepare_for_splink
    splink_df = prepare_for_splink(patients_df)

    splink_cfg = {
        "splink": {
            "predict_threshold": 0.01,
            "auto_match_probability": 0.95,
            "auto_reject_probability": 0.05,
        }
    }

    # Create and train
    t0 = time.time()
    linker, _ = create_linker(splink_df, splink_cfg)
    train_model(linker)
    t_train = time.time() - t0

    # Predict
    t0 = time.time()
    all_predictions = predict_matches(linker, splink_cfg)
    t_predict = time.time() - t0

    auto_matches, gray_zone, all_preds = classify_predictions(all_predictions, splink_cfg)

    # Build pair -> probability mapping
    pairs_with_probs = []
    for _, row in all_preds.iterrows():
        pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        pairs_with_probs.append((pair, row["match_probability"]))

    # Threshold sweep for probability
    prob_thresholds = [round(x, 2) for x in np.arange(0.01, 1.00, 0.01)]
    sweep = threshold_sweep(pairs_with_probs, true_pairs, prob_thresholds)
    best = max(sweep, key=lambda x: x["f1"])

    # Auto-match at 0.95
    auto_match_pairs = {pair for pair, p in pairs_with_probs if p >= 0.95}
    auto_match_metrics = compute_prf1(auto_match_pairs, true_pairs)

    gray_zone_count = sum(1 for _, p in pairs_with_probs if 0.05 <= p < 0.95)

    # Blocking recall (all predictions above threshold 0.01 = blocking output)
    all_pred_pairs = {pair for pair, _ in pairs_with_probs}
    true_found = sum(1 for p in true_pairs if p in all_pred_pairs)
    blocking_recall = true_found / len(true_pairs) if true_pairs else 0

    # Score distributions
    match_probs = [p for pair, p in pairs_with_probs if pair in true_pairs]
    nonmatch_probs = [p for pair, p in pairs_with_probs if pair not in true_pairs]

    return {
        "method": "splink",
        "train_time_s": round(t_train, 2),
        "predict_time_s": round(t_predict, 2),
        "candidate_pairs": len(all_preds),
        "blocking_recall": blocking_recall,
        "auto_match_count_0.95": len(auto_match_pairs),
        "auto_match_precision_0.95": round(auto_match_metrics["precision"], 4),
        "auto_match_recall_0.95": round(auto_match_metrics["recall"], 4),
        "auto_match_f1_0.95": round(auto_match_metrics["f1"], 4),
        "gray_zone_count_0.05_0.95": gray_zone_count,
        "best_threshold": best["threshold"],
        "best_f1": round(best["f1"], 4),
        "best_precision": round(best["precision"], 4),
        "best_recall": round(best["recall"], 4),
        "prob_mean_matches": round(np.mean(match_probs), 4) if match_probs else None,
        "prob_mean_nonmatches": round(np.mean(nonmatch_probs), 4) if nonmatch_probs else None,
        "prob_std_matches": round(np.std(match_probs), 4) if match_probs else None,
        "prob_std_nonmatches": round(np.std(nonmatch_probs), 4) if nonmatch_probs else None,
        "sweep": sweep,
    }


if __name__ == "__main__":
    augmented_dir = str(_project_root / "output" / "augmented")

    # Find run dir
    run_dirs = sorted(Path(augmented_dir).glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directory in {augmented_dir}")
    run_dir = run_dirs[-1]
    logger.info(f"Using run directory: {run_dir}")

    # Load data
    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)
    logger.info(f"Loaded {len(patients_df)} records from {patients_df['facility_id'].nunique()} facilities")

    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    record_id_mapping = patients_df[["record_id", "facility_id", "id"]]
    true_pairs = generate_true_pairs_from_ground_truth(ground_truth_df, record_id_mapping)
    logger.info(f"True match pairs: {len(true_pairs)}")

    patients_indexed = patients_df.set_index("record_id")

    old_cfg = {
        "blocking": {"strategy": "aggressive_multipass"},
        "comparison": {"ssn_fuzzy": False, "birthdate_fuzzy": False},
        "classification": {
            "method": "tiered",
            "auto_reject_threshold": 4.0,
            "auto_match_threshold": 6.0,
            "single_threshold": 5.60,
        },
    }

    # Run both methods
    rl_results = run_recordlinkage(patients_df, patients_indexed, true_pairs, old_cfg)
    splink_results = run_splink(patients_df, true_pairs, old_cfg)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON: recordlinkage vs Splink v4")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'recordlinkage':>15} {'Splink':>15}")
    print("-" * 70)
    print(f"{'Candidate pairs':<40} {rl_results['candidate_pairs']:>15,} {splink_results['candidate_pairs']:>15,}")
    print(f"{'Blocking recall':<40} {rl_results['blocking_recall']:>15.4f} {splink_results['blocking_recall']:>15.4f}")
    print()

    print(f"{'Auto-match count':<40} {rl_results['auto_match_count_6.0']:>15,} {splink_results['auto_match_count_0.95']:>15,}")
    print(f"{'Auto-match precision':<40} {rl_results['auto_match_precision_6.0']:>15.4f} {splink_results['auto_match_precision_0.95']:>15.4f}")
    print(f"{'Auto-match recall':<40} {rl_results['auto_match_recall_6.0']:>15.4f} {splink_results['auto_match_recall_0.95']:>15.4f}")
    print(f"{'Auto-match F1':<40} {rl_results['auto_match_f1_6.0']:>15.4f} {splink_results['auto_match_f1_0.95']:>15.4f}")
    print()

    print(f"{'Gray zone count':<40} {rl_results['gray_zone_count_4.0_6.0']:>15,} {splink_results['gray_zone_count_0.05_0.95']:>15,}")
    print()

    print(f"{'Best single-threshold':<40} {rl_results['best_threshold']:>15} {splink_results['best_threshold']:>15}")
    print(f"{'Best F1':<40} {rl_results['best_f1']:>15.4f} {splink_results['best_f1']:>15.4f}")
    print(f"{'Best precision':<40} {rl_results['best_precision']:>15.4f} {splink_results['best_precision']:>15.4f}")
    print(f"{'Best recall':<40} {rl_results['best_recall']:>15.4f} {splink_results['best_recall']:>15.4f}")
    print()

    print(f"{'Mean score (matches)':<40} {rl_results['score_mean_matches']:>15} {splink_results['prob_mean_matches']:>15}")
    print(f"{'Mean score (non-matches)':<40} {rl_results['score_mean_nonmatches']:>15} {splink_results['prob_mean_nonmatches']:>15}")
    print(f"{'StdDev score (matches)':<40} {rl_results['score_std_matches']:>15} {splink_results['prob_std_matches']:>15}")
    print(f"{'StdDev score (non-matches)':<40} {rl_results['score_std_nonmatches']:>15} {splink_results['prob_std_nonmatches']:>15}")

    # Save full results
    out_dir = _project_root / "output" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove sweep from summary (too large for display)
    rl_summary = {k: v for k, v in rl_results.items() if k != "sweep"}
    sp_summary = {k: v for k, v in splink_results.items() if k != "sweep"}

    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump({"recordlinkage": rl_summary, "splink": sp_summary}, f, indent=2)

    # Save sweeps as CSV for analysis
    rl_sweep_df = pd.DataFrame(rl_results["sweep"])
    rl_sweep_df.to_csv(out_dir / "recordlinkage_sweep.csv", index=False)

    sp_sweep_df = pd.DataFrame(splink_results["sweep"])
    sp_sweep_df.to_csv(out_dir / "splink_sweep.csv", index=False)

    logger.info(f"\nResults saved to {out_dir}/")

    # Error analysis: what does each method get wrong?
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # recordlinkage auto-match errors at 6.0
    rl_auto_pairs = {pair for pair, s in [(tuple(sorted([idx[0], idx[1]])), features_row)
                      for idx, features_row in zip(
                          # Re-extract from sweep data isn't easy, use the stored count
                          [], [])] if s >= 6.0}

    # Re-run to get actual pair sets for error analysis
    # recordlinkage at best threshold
    rl_best_t = rl_results["best_threshold"]
    # splink at best threshold
    sp_best_t = splink_results["best_threshold"]

    print(f"\nrecordlinkage at best threshold ({rl_best_t}):")
    rl_best = next(s for s in rl_results["sweep"] if s["threshold"] == rl_best_t)
    print(f"  TP={rl_best['tp']}, FP={rl_best['fp']}, FN={rl_best['fn']}")

    print(f"\nSplink at best threshold ({sp_best_t}):")
    sp_best = next(s for s in splink_results["sweep"] if s["threshold"] == sp_best_t)
    print(f"  TP={sp_best['tp']}, FP={sp_best['fp']}, FN={sp_best['fn']}")

    print(f"\nrecordlinkage auto-match (>= 6.0):")
    rl_6 = next(s for s in rl_results["sweep"] if s["threshold"] == 6.0)
    print(f"  TP={rl_6['tp']}, FP={rl_6['fp']}, FN={rl_6['fn']}")

    print(f"\nSplink auto-match (>= 0.95):")
    sp_95 = next(s for s in splink_results["sweep"] if s["threshold"] == 0.95)
    print(f"  TP={sp_95['tp']}, FP={sp_95['fp']}, FN={sp_95['fn']}")

    print("\nDone!")
