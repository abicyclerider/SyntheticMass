#!/usr/bin/env python3
"""Merge runs from a per-training MLflow database into a persistent history database.

Each RunPod training run produces a fresh mlflow.db with a single run.
This script copies those runs into a long-lived mlflow_history.db so all
training runs can be compared side-by-side in the MLflow UI.

Usage:
    python merge_mlflow_runs.py <source_db> <dest_db>
    python merge_mlflow_runs.py output/training/train/mlflow.db mlflow_history.db
"""

import sys

import mlflow
from mlflow.tracking import MlflowClient


def merge(src_path: str, dst_path: str) -> None:
    # Read all runs from source
    mlflow.set_tracking_uri(f"sqlite:///{src_path}")
    src = MlflowClient()

    runs_data = []
    for exp in src.search_experiments():
        if exp.name == "Default":
            continue
        for run in src.search_runs(exp.experiment_id):
            # Get full metric history for each key
            metrics = {}
            for key in run.data.metrics:
                history = src.get_metric_history(run.info.run_id, key)
                metrics[key] = [(m.timestamp, m.value, m.step) for m in history]

            runs_data.append(
                {
                    "exp_name": exp.name,
                    "run_name": run.info.run_name,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "status": run.info.status,
                    "metrics": metrics,
                    "params": dict(run.data.params),
                    "tags": {
                        k: v
                        for k, v in run.data.tags.items()
                        if not k.startswith("mlflow.")
                    },
                }
            )

    if not runs_data:
        print("  No runs found in source database.")
        return

    # Write to destination
    mlflow.set_tracking_uri(f"sqlite:///{dst_path}")
    dst = MlflowClient()

    for d in runs_data:
        # Create experiment if needed
        exp = dst.get_experiment_by_name(d["exp_name"])
        if exp is None:
            exp_id = dst.create_experiment(d["exp_name"])
        else:
            exp_id = exp.experiment_id

        run = dst.create_run(
            exp_id, run_name=d["run_name"], start_time=d["start_time"]
        )
        rid = run.info.run_id

        for k, v in d["params"].items():
            try:
                dst.log_param(rid, k, v[:8000])
            except Exception:
                pass

        for key, points in d["metrics"].items():
            for ts, val, step in points:
                dst.log_metric(rid, key, val, timestamp=ts, step=step)

        for k, v in d["tags"].items():
            try:
                dst.set_tag(rid, k, v[:8000])
            except Exception:
                pass

        dst.set_terminated(rid, status=d["status"], end_time=d["end_time"])
        print(f"  Merged run: {d['run_name']} -> {rid}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source_db> <dest_db>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
