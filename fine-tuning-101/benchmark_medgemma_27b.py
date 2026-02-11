#!/usr/bin/env python3
"""
Benchmark MedGemma 27B on entity resolution test pairs.

Self-contained script (no shared/ imports) designed to run on a RunPod GPU.
Downloads test pairs from HF Hub and the model from HF, runs inference,
computes metrics, and saves detailed results.

Usage:
    python benchmark_medgemma_27b.py                    # bf16, A100 80GB
    python benchmark_medgemma_27b.py --quantize-4bit    # 4-bit, fits ~24GB
    python benchmark_medgemma_27b.py --quantize-8bit    # 8-bit, fits ~40GB
    python benchmark_medgemma_27b.py --system-prompt "You are a helpful medical assistant."
    python benchmark_medgemma_27b.py --limit 5          # quick smoke test
"""

import argparse
import json
import re
import time

import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "google/medgemma-27b-text-it"
DATASET_REPO = "abicyclerider/entity-resolution-pairs"


def extract_answer(text):
    """Strip MedGemma thinking tokens from response.

    MedGemma may output <unused94>thought...<unused95> before the actual answer.
    Safety net — 27B likely doesn't use these, unlike 4B.
    """
    cleaned = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<unused94>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_prediction(raw_response):
    """Parse True/False from model response."""
    cleaned = extract_answer(raw_response).lower()
    if "true" in cleaned:
        return True
    elif "false" in cleaned:
        return False
    return None


def get_gpu_info():
    """Return GPU info dict."""
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "gpu_count": torch.cuda.device_count(),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MedGemma 27B on entity resolution")
    parser.add_argument("--quantize-4bit", action="store_true",
                        help="Load in 4-bit (bitsandbytes, fits ~24GB GPU)")
    parser.add_argument("--quantize-8bit", action="store_true",
                        help="Load in 8-bit (bitsandbytes, fits ~40GB GPU)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt (default: none, for consistency with prior benchmarks)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test pairs (for quick smoke tests)")
    args = parser.parse_args()

    # --- GPU info ---
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('gpu_name', 'N/A')} ({gpu_info.get('gpu_memory_gb', 'N/A')} GB)")

    # --- Load dataset ---
    print(f"\nLoading test set from {DATASET_REPO}...")
    dataset = load_dataset(DATASET_REPO)

    test_prompts = []
    test_labels = []
    for example in dataset["test"]:
        test_prompts.append(example["messages"][0]["content"])
        test_labels.append(example["messages"][1]["content"] == "True")

    if args.limit:
        test_prompts = test_prompts[: args.limit]
        test_labels = test_labels[: args.limit]

    n_match = sum(test_labels)
    n_non = len(test_labels) - n_match
    print(f"Test set: {len(test_labels)} pairs ({n_match} match + {n_non} non-match)")

    # --- Load model ---
    print(f"\nLoading {MODEL_ID}...")
    load_start = time.time()

    quantization_config = None
    quant_label = "bf16"
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        quant_label = "4bit"
    elif args.quantize_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        quant_label = "8bit"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s (quantization: {quant_label})")

    # --- Run inference ---
    print(f"\nRunning inference on {len(test_prompts)} pairs...")
    predictions = []
    raw_responses = []
    pair_times = []
    unparseable_indices = []

    for i, (prompt, label) in enumerate(
        tqdm(zip(test_prompts, test_labels), total=len(test_prompts))
    ):
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=8192,
        ).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        elapsed = time.time() - t0
        pair_times.append(elapsed)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True,
        ).strip()
        raw_responses.append(response)

        pred = parse_prediction(response)
        predictions.append(pred)
        if pred is None:
            unparseable_indices.append(i)

        if (i + 1) % 50 == 0:
            valid = [p for p in predictions if p is not None]
            valid_labels = [l for p, l in zip(predictions, test_labels) if p is not None]
            acc = accuracy_score(valid_labels, valid) if valid else 0
            avg_time = sum(pair_times) / len(pair_times)
            print(
                f"  [{i + 1}/{len(test_prompts)}] acc={acc:.3f}, "
                f"avg={avg_time:.1f}s/pair, unparseable={len(unparseable_indices)}"
            )

    # --- Compute metrics (exclude unparseable) ---
    valid_preds = [p for p in predictions if p is not None]
    valid_labels = [l for p, l in zip(predictions, test_labels) if p is not None]

    metrics = {
        "accuracy": round(accuracy_score(valid_labels, valid_preds), 4),
        "precision": round(precision_score(valid_labels, valid_preds, zero_division=0), 4),
        "recall": round(recall_score(valid_labels, valid_preds, zero_division=0), 4),
        "f1": round(f1_score(valid_labels, valid_preds, zero_division=0), 4),
    }
    cm = confusion_matrix(valid_labels, valid_preds).tolist()
    total_inference_time = sum(pair_times)

    # --- Print results ---
    print(f"\n{'=' * 60}")
    print(f"MedGemma 27B ({quant_label}) — Entity Resolution Results")
    print(f"{'=' * 60}")
    print(f"Parseable: {len(valid_preds)}/{len(predictions)} ({len(unparseable_indices)} unparseable)")
    for m, v in metrics.items():
        print(f"  {m:>10s}: {v:.3f}")
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\nTiming:")
    print(f"  Model load:  {load_time:.1f}s")
    print(f"  Inference:   {total_inference_time:.1f}s "
          f"({total_inference_time / len(test_prompts):.1f}s/pair)")
    print(f"  Total:       {load_time + total_inference_time:.1f}s")

    # --- Comparison table ---
    prior = {
        "Gemma 1B (base)":       {"acc": 0.527, "prec": 0.523, "rec": 0.675, "f1": 0.589,
                                   "notes": "float32, transformers"},
        "Gemma 1B (fine-tuned)": {"acc": 0.571, "prec": 0.544, "rec": 0.917, "f1": 0.683,
                                   "notes": "LoRA, 3 epochs"},
        "Claude Opus (ceiling)": {"acc": 0.940, "prec": 0.919, "rec": 0.953, "f1": 0.936,
                                   "notes": "Strategy D summaries"},
    }
    print(f"\n{'Model':<25s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  Notes")
    print("-" * 80)
    for name, m in prior.items():
        print(f"{name:<25s}  {m['acc']:>6.3f}  {m['prec']:>6.3f}  "
              f"{m['rec']:>6.3f}  {m['f1']:>6.3f}  {m['notes']}")
    print(f"{'MedGemma 27B':<25s}  {metrics['accuracy']:>6.3f}  {metrics['precision']:>6.3f}  "
          f"{metrics['recall']:>6.3f}  {metrics['f1']:>6.3f}  {quant_label}, RunPod")

    # --- Save results ---
    results = {
        "model": MODEL_ID,
        "quantization": quant_label,
        "system_prompt": args.system_prompt,
        "gpu": gpu_info,
        "dataset": DATASET_REPO,
        "n_test_pairs": len(test_prompts),
        "n_parseable": len(valid_preds),
        "n_unparseable": len(unparseable_indices),
        "metrics": metrics,
        "confusion_matrix": cm,
        "timing": {
            "model_load_s": round(load_time, 1),
            "total_inference_s": round(total_inference_time, 1),
            "avg_per_pair_s": round(total_inference_time / len(test_prompts), 2),
        },
        "per_pair": [
            {
                "index": i,
                "true_label": test_labels[i],
                "prediction": predictions[i],
                "raw_response": raw_responses[i],
                "time_s": round(pair_times[i], 2),
            }
            for i in range(len(test_prompts))
        ],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
