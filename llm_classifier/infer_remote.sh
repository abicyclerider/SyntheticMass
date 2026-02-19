#!/usr/bin/env bash
# Autonomous remote inference: launch pod → pod infers + pushes to HF Hub → collect results later.
#
# Auto-detects phase via a .state/infer_launch_timestamp file:
#   1. No timestamp file   → LAUNCH: upload input, launch pod, exit 1
#   2. Timestamp + HF Hub NOT updated → NOT READY: print status, exit 1
#   3. Timestamp + HF Hub updated     → COLLECT: download + validate predictions, exit 0
#
# Usage: ./infer_remote.sh [--gpu-type "GPU NAME"] <input_file> <output_file>
#        ./infer_remote.sh --local <input_file> <output_file>
#
# DVC workflow:
#   dvc repro infer          # launches pod, exits 1 (expected)
#   # close laptop, check HF Hub for results
#   dvc repro infer          # collects results, exits 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_helpers.sh"

# Note: HF Hub repos use "grey" spelling (external resource, can't rename)
HF_INPUT_REPO="abicyclerider/grey-zone-pairs"
HF_OUTPUT_REPO="abicyclerider/grey-zone-predictions"
GPU_TYPE="NVIDIA GeForce RTX 4090"
BATCH_SIZE=""
LOCAL_MODE=false

# --- Parse args ---
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --local) LOCAL_MODE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--local | --gpu-type \"GPU NAME\"] <input_file> <output_file>"
    exit 1
fi
INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# --- Local mode: run inference directly on this machine (MPS/CUDA) ---
if [[ "$LOCAL_MODE" == "true" ]]; then
    echo "Running inference locally..."
    echo "Input: $INPUT_FILE"
    mkdir -p "$(dirname "$OUTPUT_FILE")"

    LOCAL_BATCH_ARGS=""
    if [[ -n "$BATCH_SIZE" ]]; then
        LOCAL_BATCH_ARGS="--batch-size $BATCH_SIZE"
    fi
    if ! python3 "$SCRIPT_DIR/infer_classifier.py" \
        --input-file "$INPUT_FILE" \
        --output-file "$OUTPUT_FILE" \
        --no-quantize $LOCAL_BATCH_ARGS; then
        echo "ERROR: Local inference failed."
        exit 1
    fi

    if [[ ! -f "$OUTPUT_FILE" ]]; then
        echo "ERROR: Output file was not created: $OUTPUT_FILE"
        exit 1
    fi

    echo ""
    echo "Done. Predictions saved to $OUTPUT_FILE"
    exit 0
fi

# --- Remote mode: RunPod inference via HF Hub ---

check_python_deps datasets pandas huggingface_hub
read_credentials
mkdir -p "$STATE_DIR"

TIMESTAMP_FILE="$STATE_DIR/infer_launch_timestamp"

INPUT_ROWS=$(INPUT_FILE="$INPUT_FILE" python3 -c "import os, pandas as pd; print(len(pd.read_parquet(os.environ['INPUT_FILE'])))")

# =============================================================================
# Phase detection
# =============================================================================

if [[ -f "$TIMESTAMP_FILE" ]]; then
    BEFORE_TS=$(cat "$TIMESTAMP_FILE")
    echo "Previous launch detected (timestamp: $BEFORE_TS)"
    echo "Checking if predictions repo was updated..."

    if check_hub_updated "$HF_OUTPUT_REPO" "$BEFORE_TS" "dataset"; then
        # --- COLLECT MODE ---
        echo "Predictions repo updated — collecting results."
        echo ""

        # Download predictions from HF Hub
        echo "=== Download predictions from HF Hub ($HF_OUTPUT_REPO) ==="
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        if ! HF_TOKEN="$HF_TOKEN" OUTPUT_FILE="$OUTPUT_FILE" HF_REPO="$HF_OUTPUT_REPO" python3 -c "
import os
from datasets import load_dataset
ds = load_dataset(os.environ['HF_REPO'], split='train', download_mode='force_redownload')
df = ds.to_pandas()
df.to_parquet(os.environ['OUTPUT_FILE'], index=False)
print(f'  Downloaded {len(df)} rows')
"; then
            echo "ERROR: Failed to download predictions from HF Hub ($HF_OUTPUT_REPO)."
            echo "  The pod finished but predictions may not have been uploaded."
            echo "  Check pod logs for inference errors."
            exit 1
        fi

        if [[ ! -f "$OUTPUT_FILE" ]]; then
            echo "ERROR: Output file was not created: $OUTPUT_FILE"
            exit 1
        fi

        # Validate output
        echo ""
        echo "=== Validate predictions ==="
        if ! OUTPUT_FILE="$OUTPUT_FILE" INPUT_ROWS="$INPUT_ROWS" python3 -c "
import os, sys
import pandas as pd
df = pd.read_parquet(os.environ['OUTPUT_FILE'])
expected = int(os.environ['INPUT_ROWS'])
errors = []
if 'prediction' not in df.columns:
    errors.append('Missing column: prediction')
if 'confidence' not in df.columns:
    errors.append('Missing column: confidence')
if len(df) != expected:
    errors.append(f'Row count mismatch: expected {expected}, got {len(df)}')
if errors:
    for e in errors:
        print(f'  FAIL: {e}')
    sys.exit(1)
match = (df['prediction'] == 1).sum()
nonmatch = (df['prediction'] == 0).sum()
print(f'  OK: {len(df)} rows with prediction + confidence')
print(f'  Predictions: {match} match, {nonmatch} non-match')
print(f'  Mean confidence: {df[\"confidence\"].mean():.4f}')
"; then
            echo "ERROR: Prediction validation failed. See errors above."
            echo "  The predictions file may be incomplete or malformed."
            exit 1
        fi

        # Clean up state
        rm -f "$TIMESTAMP_FILE"

        echo ""
        echo "Done. Predictions saved to $OUTPUT_FILE"
        exit 0
    else
        # --- NOT READY ---
        echo ""
        echo "Predictions repo NOT updated yet — inference still in progress."
        echo ""
        echo "Check progress:"
        echo "  - Pod status: runpodctl get pod"
        echo ""
        echo "Re-run 'dvc repro infer' when inference completes."
        exit 1
    fi
fi

# =============================================================================
# LAUNCH MODE — no timestamp file exists
# =============================================================================

echo "Inference remote run"
echo "  Input:    $INPUT_FILE ($INPUT_ROWS rows)"
echo "  GPU type: $GPU_TYPE"

# Upload input to HF Hub
echo ""
echo "=== Upload input to HF Hub ($HF_INPUT_REPO) ==="
if ! HF_TOKEN="$HF_TOKEN" INPUT_FILE="$INPUT_FILE" HF_REPO="$HF_INPUT_REPO" python3 -c "
import os
from datasets import Dataset
import pandas as pd
df = pd.read_parquet(os.environ['INPUT_FILE'])
Dataset.from_pandas(df).push_to_hub(os.environ['HF_REPO'])
print(f'  Uploaded {len(df)} rows')
"; then
    echo "ERROR: Failed to upload input to HF Hub."
    echo "  Check your HF_TOKEN and network connection."
    exit 1
fi

# Record output repo timestamp before inference
BEFORE_TS=$(get_hub_timestamp "$HF_OUTPUT_REPO" "dataset")
echo "Predictions repo last modified: $BEFORE_TS"

# Save timestamp for collect phase
echo "$BEFORE_TS" > "$TIMESTAMP_FILE"

# Launch pod
POD_ID=""
BATCH_SIZE_FLAG=""
if [[ -n "$BATCH_SIZE" ]]; then
    BATCH_SIZE_FLAG="--batch-size $BATCH_SIZE"
fi
LAUNCH_CMD="\"$SCRIPT_DIR/launch_pod.sh\" infer --gpu-type \"$GPU_TYPE\" --hf-input \"$HF_INPUT_REPO\" --hf-output \"$HF_OUTPUT_REPO\" $BATCH_SIZE_FLAG"
launch_pod_with_retry "$LAUNCH_CMD" 3

echo ""
echo "Pod $POD_ID launched. It will run inference and push results to HF Hub."
echo ""
echo "Check progress:"
echo "  - Pod status: runpodctl get pod"
echo "  - Pod logs:   https://www.runpod.io/console/pods/$POD_ID/logs"
echo ""
echo "To manually stop: runpodctl stop pod $POD_ID"
echo "To terminate:     runpodctl remove pod $POD_ID"
echo ""
echo "Re-run 'dvc repro infer' to collect results once inference completes."
exit 1
