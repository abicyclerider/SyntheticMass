# Productionalizing Fine-Tuning for MedGemma Impact Challenge

Research and implementation plan for making model fine-tuning reproducible and submission-ready.

## Competition Submission Requirements

The [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) is a **Kaggle Writeup** competition (not a code/notebook competition). Each submission is a [Kaggle Writeup](https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups) containing:

| Artifact | Details |
|----------|---------|
| Video demo | 3 minutes max, demonstrating your application |
| Technical overview | 3 pages max |
| Reproducible source code | Linked GitHub or HuggingFace repo |

**Judging criteria** (Google Research, DeepMind, Health AI teams):
1. Effective use of HAI-DEF models (MedGemma required)
2. Problem importance and relevance
3. Real-world impact potential
4. Technical feasibility
5. Execution and communication quality

**Bonus prize categories** relevant to this project:
- **Novel fine-tuned model adaptations** — directly applies to our LoRA/QLoRA work
- **Edge AI** — applicable if model runs on local devices (mobile, scanners, lab instruments)

Submissions can be revised and resubmitted multiple times before the Feb 24, 2026 deadline.

## Current State

### What We Have
- **Gemma 1B generative LoRA** (`train_on_gpu.py`): 3 epochs, eval_loss 0.325, token accuracy 0.903
- **MedGemma 4B QLoRA classifier** (`train_classifier_on_gpu.py`): 3 epochs, F1 0.827, accuracy 0.903
- **Dataset**: 1568 train / 336 eval / 338 test pairs on HF Hub (`abicyclerider/entity-resolution-pairs`)
- **RunPod GPU training**: manual SSH, pip install each time, no containerization

### Problems
- Not reproducible — manual environment setup on each pod
- No experiment tracking — metrics scattered across terminal output and notebooks
- Hardcoded hyperparameters in training scripts
- Judges cannot easily reproduce results

## Design Decisions

### Why Not DVC?

[DVC (Data Version Control)](https://dvc.org/) was considered for dataset versioning but rejected:

- **HF Hub already versions the dataset** — `abicyclerider/entity-resolution-pairs` is git-backed with commit hashes. Pinning `revision` in the config gives exact reproducibility.
- **Dataset is small and stable** — ~2K pairs, deterministically generated from Synthea via `prepare_dataset.py`. Not a large-data management problem.
- **Adds complexity for judges** — another tool to install, another remote storage backend (S3/GCS) to configure, another `dvc pull` step.
- **Preprocessing is deterministic** — given the same Synthea input data + `prepare_dataset.py`, you get the same output.

**Instead**: pin the exact HF Hub dataset revision in each config file:

```yaml
dataset:
  name: abicyclerider/entity-resolution-pairs
  revision: "abc123"  # exact commit hash
```

This ensures every training run uses the same data without additional tooling. If we later start iterating heavily on the data pipeline (different summarization strategies, augmentation, pairing logic), DVC would earn its keep — but not yet.

### Why Not Kubernetes?

K8s and Kubeflow were considered but rejected for this use case:

| Option | Verdict | Reason |
|--------|---------|--------|
| Simple K8s Job | Overkill | Requires a GPU cluster; judges won't have one |
| Kubeflow Trainer | Way overkill | Complex setup (hours to days), designed for distributed multi-node training |
| Modal / Lightning AI | Viable but vendor lock-in | Simpler infra, but less portable for submission |
| **Docker + RunPod** | **Recommended** | Minimal change, fully reproducible, judges can run anywhere |

Our training runs are single-GPU, ~56 minutes, ~$0.17 — no orchestration needed.

## Implementation Plan

### 1. Dockerfile

Build a reproducible training environment based on [RunPod's best practices](https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11).

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Use devel variant — required for bitsandbytes CUDA kernel compilation
# Pin Ubuntu 22.04 for broad compatibility

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    software-properties-common build-essential wget curl git && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && rm get-pip.py && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

COPY requirements.txt /workspace/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY src/ /workspace/src/
COPY configs/ /workspace/configs/

WORKDIR /workspace

# Don't bake model weights — download from HF Hub at runtime
# Don't bake dataset — download from HF Hub at runtime

ENTRYPOINT ["python3.11", "src/train.py"]
CMD ["--config", "configs/default.yaml"]
```

Key design decisions:
- **Don't bake model weights or datasets into the image** — download from HF Hub at runtime (keeps image small, avoids license issues)
- **Use `nvidia/cuda` devel variant** — required for `bitsandbytes` to compile CUDA kernels
- **Pin everything** in `requirements.txt` (see below)

### 2. Pinned Dependencies

Update `requirements.txt` with exact versions:

```
# Core
torch==2.5.1
transformers==4.49.0
peft==0.14.0
trl==0.15.0
datasets==3.0.0
accelerate==1.0.0
bitsandbytes==0.45.0

# Tracking
mlflow>=2.18.0

# Utilities
huggingface-hub>=0.27.0
python-dotenv>=1.0.0
scikit-learn>=1.4.0
pyyaml>=6.0
```

### 3. YAML Config Files

Extract hardcoded hyperparameters into config files:

```yaml
# configs/medgemma_4b_qlora.yaml
model:
  name: google/medgemma-4b-it
  task: classification
  num_labels: 2

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: [score]
  bias: none

training:
  learning_rate: 1.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  warmup_steps: 50
  lr_scheduler_type: cosine
  gradient_checkpointing: true
  seed: 42

dataset:
  name: abicyclerider/entity-resolution-pairs
  revision: null  # set to HF Hub commit hash to pin exact version

output:
  dir: /workspace/output
  hub_model_id: abicyclerider/medgemma-4b-entity-resolution-classifier
```

```yaml
# configs/gemma_1b_lora.yaml
model:
  name: google/gemma-3-1b-it
  task: generative

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  bias: none

training:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 2048
  warmup_steps: 25
  lr_scheduler_type: cosine
  gradient_checkpointing: true
  seed: 42

dataset:
  name: abicyclerider/entity-resolution-pairs
  revision: null  # set to HF Hub commit hash to pin exact version

output:
  dir: /workspace/output
  hub_model_id: abicyclerider/gemma-1b-entity-resolution-lora
```

### 4. MLflow Integration

[MLflow has native PEFT/LoRA support](https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/). The integration requires minimal code changes.

#### In the training script

```python
import mlflow

# Set tracking URI (from env or config)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment("entity-resolution")

# In TrainingArguments — this is all you need for automatic logging:
training_args = TrainingArguments(
    report_to="mlflow",
    run_name=f"medgemma-4b-qlora-{datetime.now():%Y%m%d-%H%M}",
    ...
)

# After training, log the PEFT adapter explicitly:
with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
    mlflow.log_params(peft_config.to_dict())
    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        name="model",
    )
```

**What gets logged automatically** (via `report_to="mlflow"`):
- Training loss at each logging step
- Eval metrics (loss, F1, accuracy) per epoch
- All hyperparameters from TrainingArguments, LoraConfig, BitsAndBytesConfig

**What `log_model` saves** for PEFT:
- Only the adapter weights (~few MB), not the base model
- A reference to the HF Hub base model (repo + commit hash)
- Tokenizer and config

#### Tracking server options

| Option | Setup | Accessible from RunPod? | Cost |
|--------|-------|-------------------------|------|
| **[DagsHub](https://dagshub.com/blog/free-remote-mlflow-server/)** (recommended) | Create repo, get URI | Yes (remote) | Free |
| Local `mlflow server` | `mlflow server --host 0.0.0.0` | No (unless tunneled) | Free |
| [Docker Compose](https://mlflow.org/docs/latest/self-hosting/) (MLflow + Postgres + MinIO) | Deploy on a VPS | Yes (remote) | ~$5/mo |

**Recommendation**: Start with **DagsHub** — zero setup, free, accessible from both local machine and RunPod. Each DagsHub repo gets a free MLflow tracking server at `https://dagshub.com/<user>/<repo>.mlflow`.

### 5. Unified Training Entrypoint

Refactor `train_on_gpu.py` and `train_classifier_on_gpu.py` into a single `src/train.py` that:
- Reads config from YAML file (`--config` argument)
- Dispatches to generative (SFTTrainer) or classifier (Trainer) based on `model.task`
- Sets up MLflow tracking
- Handles HF Hub login from `.env`
- Pushes adapter to HF Hub on completion

### 6. RunPod Workflow

Replace the current manual setup:

```bash
# One-time: build and push image
docker build -t yourusername/medgemma-ft:latest .
docker push yourusername/medgemma-ft:latest

# Launch training
runpodctl create pod \
  --name "medgemma-ft" \
  --gpuType "NVIDIA L40" \
  --imageName "yourusername/medgemma-ft:latest" \
  --containerDiskSize 30 \
  --ports "22/tcp" \
  --env "HF_TOKEN=hf_..." \
  --env "MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow" \
  --env "MLFLOW_TRACKING_USERNAME=..." \
  --env "MLFLOW_TRACKING_PASSWORD=..."

# SSH in and start training
runpodctl ssh connect <podId>
python3.11 src/train.py --config configs/medgemma_4b_qlora.yaml
```

Note: since `--env` flags are unreliable on RunPod (known issue), the training script should also support loading env vars from `.env` file passed via `--env-file` or entered manually after SSH.

### 7. Proposed Directory Structure

```
fine-tuning-101/
├── Dockerfile                          # Reproducible training environment
├── .dockerignore                       # Exclude output/, .venv/, notebooks
├── docker-compose.yml                  # Optional: local MLflow server
├── requirements.txt                    # Pinned dependencies
├── .env                                # HF_TOKEN, MLFLOW_TRACKING_URI (gitignored)
├── .env.example                        # Template for .env
├── configs/
│   ├── gemma_1b_lora.yaml              # Strategy A hyperparams
│   └── medgemma_4b_qlora.yaml          # Strategy B hyperparams
├── src/
│   ├── train.py                        # Unified training entrypoint
│   ├── evaluate.py                     # Standalone evaluation on test set
│   └── prepare_dataset.py              # Dataset preparation (push to HF Hub)
├── scripts/
│   ├── train_runpod.sh                 # Launch RunPod pod + train
│   └── train_local.sh                  # Local training (MPS device)
├── output/                             # Local training output (gitignored)
├── RUNPOD_GUIDE.md                     # RunPod setup reference
├── PRODUCTIONALIZATION.md              # This document
└── README.md                           # Quick start guide
```

## Before vs After

| Aspect | Before (current) | After (proposed) |
|--------|-------------------|------------------|
| Environment setup | Manual SSH + pip install each pod | Pre-built Docker image |
| Reproducibility | Not reproducible | `docker run --gpus all` anywhere |
| Experiment tracking | None | MLflow logs every run |
| Configuration | Hardcoded in scripts | YAML config files |
| Iteration speed | ~15 min setup + 56 min train | ~2 min pod start + 56 min train |
| Judge reproducibility | Difficult | Dockerfile + README |
| Data versioning | None (latest from HF Hub) | Pinned HF Hub revision in config |
| Model artifacts | Manual HF Hub push | Auto-logged to MLflow + HF Hub |

## Implementation Order

1. **Create `configs/` with YAML files** — extract hyperparams from existing scripts
2. **Refactor to unified `src/train.py`** — consolidate two training scripts
3. **Add MLflow integration** — `report_to="mlflow"` + DagsHub setup
4. **Write `Dockerfile`** — pin all dependencies, test locally
5. **Build + push Docker image** — test on RunPod
6. **Write helper scripts** — `scripts/train_runpod.sh`, `scripts/train_local.sh`
7. **Update README** — document the full workflow for judges

## References

- [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- [Competition details (EdTech Innovation Hub)](https://www.edtechinnovationhub.com/news/google-launches-medgemma-impact-challenge-to-advance-human-centered-health-ai)
- [MLflow PEFT/QLoRA Tutorial](https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/)
- [MLflow Self-Hosting](https://mlflow.org/docs/latest/self-hosting/)
- [DagsHub Free MLflow Server](https://dagshub.com/blog/free-remote-mlflow-server/)
- [RunPod Docker + CUDA Setup](https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11)
- [RunPod Reproducible AI Guide](https://www.runpod.io/articles/guides/reproducible-ai-made-easy-versioning-data-and-tracking-experiments)
