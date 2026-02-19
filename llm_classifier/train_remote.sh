#!/usr/bin/env bash
# Autonomous remote training: launch pod → pod trains + pushes to HF Hub → collect results later.
#
# Auto-detects phase via a .state/train_launch_timestamp file:
#   1. No timestamp file   → LAUNCH: record timestamp, launch pod, exit 1
#   2. Timestamp + HF Hub NOT updated → NOT READY: print status, exit 1
#   3. Timestamp + HF Hub updated     → COLLECT: download metrics, run promotion gate, exit 0
#
# Usage: ./train_remote.sh [--gpu-type "GPU NAME"] <output_dir> [-- training_args...]
# Example: ./train_remote.sh output/training/train -- --epochs 3 --batch-size 4
#
# DVC workflow:
#   dvc repro train          # launches pod, exits 1 (expected)
#   # close laptop, check HF Hub checkpoint repo for progress
#   dvc repro train          # collects results, exits 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_helpers.sh"

ADAPTER_REPO="abicyclerider/medgemma-4b-entity-resolution-classifier"
GPU_TYPE="NVIDIA GeForce RTX 4090"

# --- Parse args ---
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--gpu-type \"GPU NAME\"] <output_dir> [-- training_args...]"
    exit 1
fi
OUTPUT_DIR="$1"
shift

# Collect training args after "--"
TRAINING_ARGS=""
if [[ "${1:-}" == "--" ]]; then
    shift
    TRAINING_ARGS="$*"
fi

# --- Setup ---
check_python_deps huggingface_hub
read_credentials
mkdir -p "$OUTPUT_DIR"
mkdir -p "$STATE_DIR"

TIMESTAMP_FILE="$STATE_DIR/train_launch_timestamp"

# =============================================================================
# Phase detection
# =============================================================================

if [[ -f "$TIMESTAMP_FILE" ]]; then
    BEFORE_TS=$(cat "$TIMESTAMP_FILE")
    echo "Previous launch detected (timestamp: $BEFORE_TS)"
    echo "Checking if adapter repo was updated..."

    if check_hub_updated "$ADAPTER_REPO" "$BEFORE_TS"; then
        # --- COLLECT MODE ---
        echo "Adapter repo updated — collecting results."
        echo ""

        # Download training_metrics.json
        echo "=== Download training metrics ==="
        if ! HF_TOKEN="$HF_TOKEN" REPO="$ADAPTER_REPO" OUTPUT="$OUTPUT_DIR/train_metrics.json" python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ['REPO'],
    filename='training_metrics.json',
    token=os.environ['HF_TOKEN'],
    force_download=True,
)
shutil.copy2(path, os.environ['OUTPUT'])
print(f'  Downloaded to {os.environ[\"OUTPUT\"]}')
"; then
            echo "ERROR: Failed to download training_metrics.json from $ADAPTER_REPO."
            echo "  The adapter may have been pushed without the metrics file."
            exit 1
        fi

        # Download MLflow database
        echo ""
        echo "=== Download MLflow database ==="
        if ! HF_TOKEN="$HF_TOKEN" REPO="$ADAPTER_REPO" OUTPUT="$OUTPUT_DIR/mlflow.db" python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ['REPO'],
    filename='mlflow.db',
    token=os.environ['HF_TOKEN'],
    force_download=True,
)
shutil.copy2(path, os.environ['OUTPUT'])
print(f'  Downloaded to {os.environ[\"OUTPUT\"]}')
"; then
            echo "WARNING: Failed to download mlflow.db (non-fatal)."
        fi

        # Merge into persistent history database
        HISTORY_DB="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)/mlflow_history.db"
        RUN_DB="$(cd "$OUTPUT_DIR" && pwd)/mlflow.db"
        if [[ -f "$RUN_DB" ]]; then
            echo ""
            echo "=== Merge into MLflow history ==="
            if python3 "$SCRIPT_DIR/merge_mlflow_runs.py" "$RUN_DB" "$HISTORY_DB"; then
                echo "  History DB: $HISTORY_DB"
            else
                echo "WARNING: Failed to merge MLflow runs (non-fatal)."
            fi
        fi

        # Promotion gate
        echo ""
        echo "=== Model promotion gate ==="
        python3 "$SCRIPT_DIR/promote_model.py" \
            "$HISTORY_DB" \
            "$OUTPUT_DIR/train_metrics.json" \
            "$OUTPUT_DIR/promotion_decision.json"

        PROMOTED=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/promotion_decision.json'))['promoted'])")
        if [[ "$PROMOTED" == "True" || "$PROMOTED" == "true" ]]; then
            echo "  Model PROMOTED — export stage will push to production."
        else
            echo "  Model NOT promoted — export stage will be skipped."
        fi

        # Clean up state
        rm -f "$TIMESTAMP_FILE"

        echo ""
        echo "Done. Training metrics saved to $OUTPUT_DIR/train_metrics.json"
        echo "View all runs: mlflow ui --backend-store-uri sqlite:///$HISTORY_DB"
        exit 0
    else
        # --- NOT READY ---
        echo ""
        echo "Adapter repo NOT updated yet — training still in progress."
        echo ""
        echo "Check progress:"
        echo "  - HF Hub checkpoint commits: https://huggingface.co/abicyclerider/medgemma-4b-er-classifier-checkpoints/commits/main"
        echo "  - Pod status: runpodctl get pod"
        echo ""
        echo "Re-run 'dvc repro train' when training completes."
        exit 1
    fi
fi

# =============================================================================
# LAUNCH MODE — no timestamp file exists
# =============================================================================

echo "Training remote run"
echo "  GPU type:   $GPU_TYPE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Train args: $TRAINING_ARGS"

# Record adapter repo timestamp before training
BEFORE_TS=$(get_hub_timestamp "$ADAPTER_REPO")
echo "Adapter repo last modified: $BEFORE_TS"

# Save timestamp for collect phase
echo "$BEFORE_TS" > "$TIMESTAMP_FILE"

# Launch pod
POD_ID=""
LAUNCH_CMD="\"$SCRIPT_DIR/launch_pod.sh\" train --gpu-type \"$GPU_TYPE\" $TRAINING_ARGS"
launch_pod_with_retry "$LAUNCH_CMD" 3

echo ""
echo "Pod $POD_ID launched. It will train autonomously and push results to HF Hub."
echo ""
echo "Check progress:"
echo "  - HF Hub checkpoint commits: https://huggingface.co/abicyclerider/medgemma-4b-er-classifier-checkpoints/commits/main"
echo "  - Pod status: runpodctl get pod"
echo "  - Pod logs:   https://www.runpod.io/console/pods/$POD_ID/logs"
echo ""
echo "To manually stop: runpodctl stop pod $POD_ID"
echo "To terminate:     runpodctl remove pod $POD_ID"
echo ""
echo "Re-run 'dvc repro train' to collect results once training completes."
exit 1
