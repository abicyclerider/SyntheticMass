# Experiment Results Summary

## Formal Experiments (200-example balanced dataset unless noted)

| Experiment | Model | Strategy | Train Acc | Val Acc | Notes |
|---|---|---|---|---|---|
| **sonnet-eval-v1** | **Sonnet 4.5** | Single-shot | **0.860** | **0.820** | Zero FP, 0 parse failures. Bayes error baseline. |
| decompose-opt-v1 | MedGemma 4B Q4 (no-think) | Decompose + optimize (3 iter) | 0.600 | 0.800 | *60-pair gray-zone dataset* — noisy, val set only 15 examples |
| balanced-decompose-opt-v1 | MedGemma 4B Q4 (no-think) | Decompose + optimize (2 iter) | 0.480 | 0.560 | Best iter predicts all NO (TP=0, FP=0) |
| baseline-q4-3iter | MedGemma 4B Q4 (no-think) | Single-shot + optimize (3 iter) | 0.311 | 0.533 | *60-pair gray-zone dataset* |
| decompose-v1 | MedGemma 4B Q4 (no-think) | Decompose (no optimize) | 0.578 | 0.400 | *60-pair gray-zone dataset* |

## Quick Evaluations (ad-hoc, small samples)

| Test | Model | Prompt | N | Correct | Notes |
|---|---|---|---|---|---|
| Baseline raw | MedGemma 4B full (with thinking) | Baseline v1 | 20 | 7/20 (35%) | Never says true. When thinking triggers, burns tokens recapping data. |
| Higher token limit | MedGemma 4B full (with thinking) | Baseline v1, max_tokens=2048 | 6 | 3/6 (50%) | Thinking completes but still always says false. |
| Think-carefully prompt | MedGemma 4B full (with thinking) | Step-by-step + "don't default to false" | 6 | 1/6 (17%) | Repetition loops on lists, hallucinated contradictions. *Worse* than no prompt engineering. |

## Key Findings

1. **Sonnet 4.5 proves the task is solvable** — 82% val accuracy with zero false positives using the simplest possible prompt. The medical history signal is sufficient for entity resolution.

2. **MedGemma 4B cannot do this task regardless of prompting strategy.** It has a strong prior toward "false" / "no match" and never learns to discriminate:
   - Without thinking: reflexively outputs `is_match: false, confidence: 0.99`
   - With thinking: either loops repetitively or recaps data until token limit, then still says false
   - With step-by-step instructions: invents false contradictions to justify "no match"
   - Decompose strategy oscillates between all-YES and all-NO across optimization iterations

3. **The earlier decompose-opt-v1 result (0.800 val) was misleading** — it used only 60 gray-zone pairs with a 15-example val set where each example was worth 6.7%. The balanced 200-example dataset revealed the true performance (~50%, i.e. random).
