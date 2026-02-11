#!/usr/bin/env bash
# =============================================================================
# RunPod MedGemma 27B Benchmark — Copy-Paste Commands
# =============================================================================
#
# NOT meant to be run as a single script. Copy-paste each section in order.
#
# Prerequisites:
#   1. Accept MedGemma license at https://huggingface.co/google/medgemma-27b-text-it
#   2. runpodctl installed (~/bin/runpodctl)
#   3. HF token with read access to gated models
#
# Estimated cost: ~$2-4 (A100 80GB @ ~$1.39-1.89/hr for ~1.5-2 hrs)
# Dataset: 6,482 test pairs @ ~0.86s/pair = ~93 min inference + ~3 min model load
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Create pod (A100 80GB, 80GB disk for model weights)
# ---------------------------------------------------------------------------
~/bin/runpodctl create pod \
  --name "medgemma-27b-bench" \
  --gpuType "NVIDIA A100-SXM4-80GB" \
  --gpuCount 1 \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
  --containerDiskSize 80 \
  --volumeSize 0 \
  --startSSH \
  --ports "22/tcp"

# ---------------------------------------------------------------------------
# 2. Wait for pod & get SSH connection info
# ---------------------------------------------------------------------------
~/bin/runpodctl get pod

# Get public IP and port for SSH (look for privatePort=22):
curl -s -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $(awk -F'"' '/apikey/{print $2}' ~/.runpod/config.toml)" \
  -d '{"query":"query { myself { pods { id name runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } desiredStatus } } }"}' \
  https://api.runpod.io/graphql | python3 -m json.tool

# Set these from the output above:
export POD_IP="<PUBLIC_IP>"
export POD_PORT="<PUBLIC_PORT>"
export POD_ID="<POD_ID>"

# ---------------------------------------------------------------------------
# 3. Install dependencies on pod
# ---------------------------------------------------------------------------
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -p $POD_PORT root@$POD_IP \
  "pip install -q transformers datasets scikit-learn accelerate tqdm && \
   pip install -q 'torch>=2.4' --index-url https://download.pytorch.org/whl/cu118 && \
   pip install -q --upgrade filelock huggingface-hub"

# ---------------------------------------------------------------------------
# 4. HuggingFace login (needed for gated MedGemma model)
# ---------------------------------------------------------------------------
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -p $POD_PORT root@$POD_IP \
  "python -c 'from huggingface_hub import login; login(token=\"hf_YOUR_TOKEN_HERE\")'"

# ---------------------------------------------------------------------------
# 5. Copy benchmark script to pod
# ---------------------------------------------------------------------------
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -P $POD_PORT \
  benchmark_medgemma_27b.py root@$POD_IP:/root/

# ---------------------------------------------------------------------------
# 6a. Smoke test first (5 pairs, ~30s)
# ---------------------------------------------------------------------------
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -p $POD_PORT root@$POD_IP \
  "cd /root && python benchmark_medgemma_27b.py \
    --system-prompt 'You are a medical entity resolution assistant. Compare the two patient records and respond with only True (same patient) or False (different patients).' \
    --limit 5 --output smoke_test.json"

# ---------------------------------------------------------------------------
# 6b. Full benchmark (6,482 pairs, ~93 min, with system prompt)
# ---------------------------------------------------------------------------
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -p $POD_PORT root@$POD_IP \
  "cd /root && nohup python benchmark_medgemma_27b.py \
    --system-prompt 'You are a medical entity resolution assistant. Compare the two patient records and respond with only True (same patient) or False (different patients).' \
    --output benchmark_results_6482.json \
    > benchmark.log 2>&1 &"

# Monitor progress:
# ssh ... "tail -f /root/benchmark.log"

# ---------------------------------------------------------------------------
# 7. Copy results back to local machine
# ---------------------------------------------------------------------------
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -P $POD_PORT \
  root@$POD_IP:/root/benchmark_results_6482.json .

# ---------------------------------------------------------------------------
# 8. Terminate pod (important — stops billing!)
# ---------------------------------------------------------------------------
~/bin/runpodctl remove pod $POD_ID
~/bin/runpodctl get pod  # confirm empty
