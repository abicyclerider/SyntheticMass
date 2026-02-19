#!/usr/bin/env bash
# Shared helper functions for RunPod wrapper scripts.
# Sourced by infer_remote.sh, train_remote.sh, export_remote.sh.
# Not executable on its own.

# --- Globals set by callers or by these functions ---
# POD_ID        - current pod ID (set by launch_pod_with_retry)
# RUNPOD_API_KEY - set by read_credentials
# HF_TOKEN       - set by read_credentials

_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$_HELPERS_DIR/.state"

# Read HF_TOKEN from .env and RUNPOD_API_KEY from ~/.runpod/config.toml.
# Sets: HF_TOKEN, RUNPOD_API_KEY
read_credentials() {
    local env_file="$_HELPERS_DIR/.env"
    if [[ ! -f "$env_file" ]]; then
        echo "ERROR: $env_file not found. Create it with HF_TOKEN=hf_..."
        exit 1
    fi
    HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' "$env_file" | tr -d '"' | tr -d "'")
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: HF_TOKEN not found in $env_file"
        exit 1
    fi

    local runpod_config="$HOME/.runpod/config.toml"
    if [[ ! -f "$runpod_config" ]]; then
        echo "ERROR: $runpod_config not found. Run: runpodctl config --apiKey YOUR_KEY"
        exit 1
    fi
    RUNPOD_API_KEY=$(sed -n 's/^apikey = "\(.*\)"/\1/p' "$runpod_config")
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        echo "ERROR: apikey not found in $runpod_config"
        exit 1
    fi
}

# Query pod status via GraphQL.
# Args: $1=pod_id
# Output: "STATUS UPTIME" on stdout (uptime=-1 if container not started)
query_pod_status() {
    local pod_id="$1"
    local status_json
    status_json=$(curl -s --max-time 15 -X POST "https://api.runpod.io/graphql" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\":\"query { pod(input: {podId: \\\"$pod_id\\\"}) { id desiredStatus runtime { uptimeInSeconds } } }\"}") || {
        echo "QUERY_FAILED -1"
        return
    }
    echo "$status_json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    pod = d.get('data', {}).get('pod')
    if pod is None:
        print('TERMINATED -1')
    else:
        status = pod.get('desiredStatus', 'UNKNOWN')
        rt = pod.get('runtime')
        uptime = rt.get('uptimeInSeconds', 0) if rt else -1
        print(f'{status} {uptime}')
except Exception:
    print('PARSE_ERROR -1')
" 2>/dev/null || echo "PARSE_ERROR -1"
}

# Launch a pod with retry on GPU unavailability.
# Args: $1=launch_cmd (must print "Pod created: <id>" on success)
#       $2=max_retries (default 3)
# Sets: POD_ID global
# Returns: 0 on success, exits 1 on failure
launch_pod_with_retry() {
    local launch_cmd="$1"
    local max_retries="${2:-3}"

    for attempt in $(seq 1 "$max_retries"); do
        echo ""
        echo "=== Launch pod (attempt $attempt/$max_retries) ==="
        local pod_output
        if pod_output=$(eval "$launch_cmd" 2>&1); then
            echo "$pod_output"
            POD_ID=$(echo "$pod_output" | sed -n 's/^Pod created: //p')
            if [[ -z "$POD_ID" ]]; then
                echo "ERROR: Failed to parse pod ID from launch output."
                exit 1
            fi
            return 0
        fi

        echo "$pod_output"
        echo ""
        echo "ERROR: Failed to create pod."
        if echo "$pod_output" | grep -qi "no gpu\|no available\|insufficient\|out of stock"; then
            echo "  GPU type appears to be unavailable on RunPod."
        else
            echo "  GPU type may not be available, or there may be an API issue."
        fi
        echo "  Check: https://www.runpod.io/console/gpu-cloud"
        if [[ $attempt -lt $max_retries ]]; then
            echo "  Retrying in 30s..."
            sleep 30
        fi
    done

    echo "  All $max_retries attempts failed."
    exit 1
}

# Check if a HF Hub repo was updated after a saved timestamp.
# Args: $1=repo_id  $2=before_timestamp (ISO format or "NONE")
# Returns: 0 if updated (or repo was new), 1 if not updated
check_hub_updated() {
    local repo_id="$1"
    local before_ts="$2"

    local after_ts
    after_ts=$(HF_TOKEN="$HF_TOKEN" REPO="$repo_id" python3 -c "
import os
from huggingface_hub import repo_info
try:
    info = repo_info(os.environ['REPO'], token=os.environ['HF_TOKEN'])
    print(info.last_modified.isoformat())
except Exception:
    print('NONE')
")

    if [[ "$before_ts" == "NONE" ]]; then
        # Repo didn't exist before — any existence means it was updated
        if [[ "$after_ts" != "NONE" ]]; then
            return 0
        fi
        return 1
    fi

    if [[ "$after_ts" != "$before_ts" ]]; then
        return 0
    fi
    return 1
}

# Get the last_modified timestamp of a HF Hub repo.
# Args: $1=repo_id
# Output: ISO timestamp or "NONE" on stdout
get_hub_timestamp() {
    local repo_id="$1"
    HF_TOKEN="$HF_TOKEN" REPO="$repo_id" python3 -c "
import os
from huggingface_hub import repo_info
try:
    info = repo_info(os.environ['REPO'], token=os.environ['HF_TOKEN'])
    print(info.last_modified.isoformat())
except Exception:
    print('NONE')
"
}

# Verify required Python packages are installed.
# Args: space-separated package names
check_python_deps() {
    local imports=""
    for pkg in "$@"; do
        if [[ -n "$imports" ]]; then
            imports="$imports, $pkg"
        else
            imports="$pkg"
        fi
    done
    python3 -c "import $imports" 2>/dev/null || {
        echo "ERROR: Required Python packages: $*"
        echo "Install: pip install $*"
        exit 1
    }
}
