#!/usr/bin/env python3
"""Quick consistency check: run MedGemma 4B on a random subset of the
(new, larger) test set and compare metrics against known baselines."""

import json
import random
import re
import time

import requests
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "medgemma:1.5-4b-q4-fast"
DATASET_REPO = "abicyclerider/entity-resolution-pairs"
N_SAMPLE = 200
SEED = 42


def extract_answer(text):
    """Strip MedGemma thinking tokens."""
    cleaned = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<unused94>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_prediction(raw):
    cleaned = extract_answer(raw).lower()
    if "true" in cleaned:
        return True
    elif "false" in cleaned:
        return False
    return None


def call_ollama(prompt, model=MODEL):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    print(f"Loading dataset from {DATASET_REPO}...")
    ds = load_dataset(DATASET_REPO)
    test = ds["test"]
    print(f"Test set: {len(test)} pairs")

    # Random sample
    random.seed(SEED)
    indices = random.sample(range(len(test)), min(N_SAMPLE, len(test)))
    print(f"Sampling {len(indices)} pairs (seed={SEED})")

    prompts = [test[i]["messages"][0]["content"] for i in indices]
    labels = [test[i]["messages"][1]["content"] == "True" for i in indices]
    n_match = sum(labels)
    print(f"  Match: {n_match}, Non-match: {len(labels) - n_match}")

    # Run inference
    print(f"\nRunning inference with {MODEL}...")
    predictions = []
    raw_responses = []
    unparseable = []
    t_start = time.time()

    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        try:
            raw = call_ollama(prompt)
        except Exception as e:
            print(f"  ERROR on pair {i}: {e}")
            raw = ""
        raw_responses.append(raw)
        pred = parse_prediction(raw)
        predictions.append(pred)
        if pred is None:
            unparseable.append(i)

        if (i + 1) % 25 == 0:
            valid = [p for p in predictions if p is not None]
            valid_l = [l for p, l in zip(predictions, labels) if p is not None]
            acc = accuracy_score(valid_l, valid) if valid else 0
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(prompts)}] acc={acc:.3f}, "
                  f"unparseable={len(unparseable)}, "
                  f"elapsed={elapsed:.0f}s ({elapsed/(i+1):.1f}s/pair)")

    total_time = time.time() - t_start

    # Compute metrics
    valid_preds = [p for p in predictions if p is not None]
    valid_labels = [l for p, l in zip(predictions, labels) if p is not None]

    metrics = {
        "accuracy": round(accuracy_score(valid_labels, valid_preds), 4),
        "precision": round(precision_score(valid_labels, valid_preds, zero_division=0), 4),
        "recall": round(recall_score(valid_labels, valid_preds, zero_division=0), 4),
        "f1": round(f1_score(valid_labels, valid_preds, zero_division=0), 4),
    }
    cm = confusion_matrix(valid_labels, valid_preds).tolist()

    # Print results
    print(f"\n{'=' * 60}")
    print(f"MedGemma 4B (q4, Ollama) — New Test Set Subset ({len(indices)} pairs)")
    print(f"{'=' * 60}")
    print(f"Parseable: {len(valid_preds)}/{len(predictions)} ({len(unparseable)} unparseable)")
    for m, v in metrics.items():
        print(f"  {m:>10s}: {v:.3f}")
    print(f"\nConfusion matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\nTiming: {total_time:.0f}s total ({total_time/len(prompts):.1f}s/pair)")

    # Comparison with known baselines (original 338-pair test set)
    print(f"\n{'=' * 60}")
    print("Comparison with original test set baselines")
    print(f"{'=' * 60}")
    prior = [
        ("Gemma 1B (base)",            0.527, 0.523, 0.675, 0.589, "original 338 pairs"),
        ("Gemma 1B (fine-tuned)",      0.571, 0.544, 0.917, 0.683, "original 338 pairs"),
        ("MedGemma 27B (no prompt)",   0.713, 0.950, 0.450, 0.610, "original 338 pairs"),
        ("MedGemma 27B (sys prompt)",  0.799, 0.848, 0.728, 0.783, "original 338 pairs"),
        ("MedGemma 4B cls (fine-tuned)", 0.903, 0.918, 0.887, 0.827, "original 338 pairs"),
        ("Claude Opus (ceiling)",      0.940, 0.919, 0.953, 0.936, "original 338 pairs"),
    ]
    print(f"{'Model':<30s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  Dataset")
    print("-" * 90)
    for name, acc, prec, rec, f1, note in prior:
        print(f"{name:<30s}  {acc:>6.3f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {note}")
    print(f"{'MedGemma 4B q4 (THIS RUN)':<30s}  {metrics['accuracy']:>6.3f}  "
          f"{metrics['precision']:>6.3f}  {metrics['recall']:>6.3f}  {metrics['f1']:>6.3f}  "
          f"new test, {len(indices)} pairs")

    # Save results
    results = {
        "model": MODEL,
        "dataset": DATASET_REPO,
        "n_sampled": len(indices),
        "n_parseable": len(valid_preds),
        "seed": SEED,
        "metrics": metrics,
        "confusion_matrix": cm,
        "timing_s": round(total_time, 1),
    }
    out_path = "consistency_check_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
