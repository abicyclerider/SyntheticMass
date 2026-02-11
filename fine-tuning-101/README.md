# Fine-Tuning 101: Local LoRA on Apple Silicon

A self-contained tutorial for fine-tuning Google's Gemma 3 1B model using LoRA on a MacBook Pro M3 Pro (18GB RAM). Runs end-to-end in ~20 minutes.

## Prerequisites

- Python 3.11+
- HuggingFace account with [Gemma model access](https://huggingface.co/google/gemma-3-1b-it) approved
- HuggingFace access token ([create one here](https://huggingface.co/settings/tokens))

## Setup

```bash
cd fine-tuning-101
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

## Run

```bash
jupyter notebook fine_tuning_101.ipynb
```

Run cells top-to-bottom. Training takes ~15-20 minutes on M3 Pro.

## What You'll Learn

1. Loading a gated model from HuggingFace
2. What LoRA is and why it makes fine-tuning practical
3. Formatting data for instruction tuning (chat templates)
4. Training with SFTTrainer
5. Comparing base vs fine-tuned model outputs
6. Saving and loading LoRA adapters
