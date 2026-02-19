#!/usr/bin/env bash
# Autonomous remote export: launch pod → pod merges LoRA + pushes to HF Hub → collect results later.
#
# Auto-detects phase via a .state/export_launch_timestamp file:
#   1. No timestamp file   → LAUNCH: launch pod, exit 1
#   2. Timestamp + HF Hub NOT updated → NOT READY: print status, exit 1
#   3. Timestamp + HF Hub updated     → COLLECT: download export info, exit 0
#
# Usage: ./export_remote.sh [--gpu-type "GPU NAME"] <output_dir>
# Example: ./export_remote.sh output/training/export
#
# DVC workflow:
#   dvc repro export         # launches pod, exits 1 (expected)
#   dvc repro export         # collects results, exits 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_helpers.sh"

OUTPUT_REPO="abicyclerider/medgemma-4b-entity-resolution-text-only"
GPU_TYPE="NVIDIA GeForce RTX 4090"

# --- Parse args ---
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--gpu-type \"GPU NAME\"] <output_dir>"
    exit 1
fi
OUTPUT_DIR="$1"

# --- Check promotion decision (applies to both launch and collect) ---
TRAIN_DIR="$(dirname "$OUTPUT_DIR")/train"
PROMOTION_FILE="$TRAIN_DIR/promotion_decision.json"
if [[ -f "$PROMOTION_FILE" ]]; then
    PROMOTED=$(python3 -c "import json; print(json.load(open('$PROMOTION_FILE')).get('promoted', True))")
    if [[ "$PROMOTED" == "False" || "$PROMOTED" == "false" ]]; then
        echo "=== Model not promoted — skipping export ==="
        mkdir -p "$OUTPUT_DIR"
        echo '{"skipped": true, "reason": "model_not_promoted"}' > "$OUTPUT_DIR/export_info.json"
        exit 0
    fi
fi

# --- Setup ---
check_python_deps huggingface_hub
read_credentials
mkdir -p "$OUTPUT_DIR"
mkdir -p "$STATE_DIR"

TIMESTAMP_FILE="$STATE_DIR/export_launch_timestamp"

# =============================================================================
# Phase detection
# =============================================================================

if [[ -f "$TIMESTAMP_FILE" ]]; then
    BEFORE_TS=$(cat "$TIMESTAMP_FILE")
    echo "Previous launch detected (timestamp: $BEFORE_TS)"
    echo "Checking if merged model repo was updated..."

    if check_hub_updated "$OUTPUT_REPO" "$BEFORE_TS"; then
        # --- COLLECT MODE ---
        echo "Merged model repo updated — collecting results."
        echo ""

        # Download export_info.json
        echo "=== Download export info ==="
        if ! HF_TOKEN="$HF_TOKEN" REPO="$OUTPUT_REPO" OUTPUT="$OUTPUT_DIR/export_info.json" python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ['REPO'],
    filename='export_info.json',
    token=os.environ['HF_TOKEN'],
    force_download=True,
)
shutil.copy2(path, os.environ['OUTPUT'])
print(f'  Downloaded to {os.environ[\"OUTPUT\"]}')
"; then
            echo "ERROR: Failed to download export_info.json from $OUTPUT_REPO."
            echo "  The model may have been pushed without the info file."
            exit 1
        fi

        # Clean up state
        rm -f "$TIMESTAMP_FILE"

        echo ""
        echo "Done. Export info saved to $OUTPUT_DIR/export_info.json"
        exit 0
    else
        # --- NOT READY ---
        echo ""
        echo "Merged model repo NOT updated yet — export still in progress."
        echo ""
        echo "Check progress:"
        echo "  - Pod status: runpodctl get pod"
        echo ""
        echo "Re-run 'dvc repro export' when export completes."
        exit 1
    fi
fi

# =============================================================================
# LAUNCH MODE — no timestamp file exists
# =============================================================================

echo "Export remote run"
echo "  GPU type:   $GPU_TYPE"
echo "  Output dir: $OUTPUT_DIR"

# Record merged model repo timestamp before export
BEFORE_TS=$(get_hub_timestamp "$OUTPUT_REPO")
echo "Merged model repo last modified: $BEFORE_TS"

# Save timestamp for collect phase
echo "$BEFORE_TS" > "$TIMESTAMP_FILE"

# Launch pod
POD_ID=""
LAUNCH_CMD="\"$SCRIPT_DIR/launch_pod.sh\" export --gpu-type \"$GPU_TYPE\" --container-disk 50"
launch_pod_with_retry "$LAUNCH_CMD" 3

echo ""
echo "Pod $POD_ID launched. It will merge LoRA + push to HF Hub."
echo ""
echo "Check progress:"
echo "  - Pod status: runpodctl get pod"
echo "  - Pod logs:   https://www.runpod.io/console/pods/$POD_ID/logs"
echo ""
echo "To manually stop: runpodctl stop pod $POD_ID"
echo "To terminate:     runpodctl remove pod $POD_ID"
echo ""
echo "Re-run 'dvc repro export' to collect results once export completes."
exit 1
