"""
Claude-powered prompt optimization for medical record entity resolution.

Uses Claude (via claude-agent-sdk) to iteratively analyze MedGemma evaluation
errors and generate improved prompts. Compares results against the DSPy MIPROv2
baseline.

Usage:
    cd claude-prompt-optimizer
    python optimize.py [--config config.yaml] [--iterations 8]
"""

import sys
import os
import re
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import anyio
import yaml
import pandas as pd
from openai import OpenAI

# Add project root to path so we can import shared modules
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add llm-entity-resolution to path for its utilities
_llm_er_root = str(Path(__file__).resolve().parent.parent / "llm-entity-resolution")
if _llm_er_root not in sys.path:
    sys.path.insert(0, _llm_er_root)

from shared.data_loader import load_facility_patients, get_run_directory
from shared.ground_truth import load_ground_truth, add_record_ids_to_ground_truth
from shared.medical_records import load_medical_records
from src.summarize import summarize_patient_records
from src.utils import extract_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# MedGemma response parsing
# ---------------------------------------------------------------------------

def parse_medgemma_response(raw_text: str) -> tuple:
    """
    Multi-stage parser for MedGemma output.

    Returns:
        (is_match: bool|None, confidence: float|None, parse_error: str|None)
    """
    # Strip thinking tokens
    text = extract_answer(raw_text)
    text_lower = text.lower()

    # Stage 1: look for is_match: true/false
    is_match = None
    match_pattern = re.search(r'is_match\s*[:=]\s*(true|false)', text_lower)
    if match_pattern:
        is_match = match_pattern.group(1) == 'true'
    else:
        # Fallback: look for standalone true/false or yes/no
        if re.search(r'\b(true|yes)\b', text_lower) and not re.search(r'\b(false|no)\b', text_lower):
            is_match = True
        elif re.search(r'\b(false|no)\b', text_lower) and not re.search(r'\b(true|yes)\b', text_lower):
            is_match = False

    # Stage 2: look for confidence: 0.XX
    confidence = None
    conf_pattern = re.search(r'confidence\s*[:=]\s*(0?\.\d+|1\.0|0\.0)', text_lower)
    if conf_pattern:
        confidence = float(conf_pattern.group(1))
    else:
        # Fallback: find any float 0.0-1.0
        floats = re.findall(r'\b(0\.\d+|1\.0)\b', text)
        if floats:
            confidence = float(floats[0])

    parse_error = None
    if is_match is None:
        parse_error = "Could not parse is_match"
    if confidence is None:
        if parse_error:
            parse_error += "; could not parse confidence"
        else:
            parse_error = "Could not parse confidence"

    return is_match, confidence, parse_error


# ---------------------------------------------------------------------------
# Decompose strategy: section parsing + sub-question helpers
# ---------------------------------------------------------------------------

# Section headers used by summarize.py (see _summarize_* functions)
_SECTION_PATTERNS = {
    'conditions': r'CONDITIONS\s*(?:\(active/historical\))?:',
    'medications': r'MEDICATIONS\s*(?:\(current/past\))?:',
    'encounters': r'ENCOUNTERS\s*(?:\(summarized\))?:',
    'observations': r'KEY OBSERVATIONS:',
    'procedures': r'PROCEDURES:',
    'immunizations': r'IMMUNIZATIONS:',
    'allergies': r'ALLERGIES:',
    'imaging': r'IMAGING:',
    'devices': r'DEVICES:',
    'care_plans': r'CARE PLANS:',
}


def extract_sections(summary: str) -> dict[str, str]:
    """
    Parse a patient summary into named sections.

    Splits on known headers from summarize.py. Returns dict with keys like
    'conditions', 'medications', 'encounters', plus 'full' for the entire text.
    """
    sections = {'full': summary}

    # Find all section start positions
    found = []
    for key, pattern in _SECTION_PATTERNS.items():
        match = re.search(pattern, summary)
        if match:
            found.append((match.start(), match.end(), key))

    # Sort by position in the text
    found.sort(key=lambda x: x[0])

    for i, (start, header_end, key) in enumerate(found):
        # Section runs from header start to next section start (or end of text)
        if i + 1 < len(found):
            end = found[i + 1][0]
        else:
            end = len(summary)
        sections[key] = summary[start:end].strip()

    return sections


def parse_yes_no(raw_text: str) -> bool | None:
    """
    Simple yes/no parser for sub-question responses.

    Strips thinking tokens, then looks for yes/no/true/false.
    Returns True, False, or None if ambiguous.
    """
    text = extract_answer(raw_text).lower()

    has_yes = bool(re.search(r'\b(yes|true)\b', text))
    has_no = bool(re.search(r'\b(no|false)\b', text))

    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def call_sub_question(prompt: str, ollama_client: OpenAI,
                      model_name: str, temperature: float,
                      max_tokens: int) -> tuple[bool | None, str]:
    """
    Call MedGemma with a sub-question, return parsed yes/no + raw response.
    """
    try:
        response = ollama_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"MedGemma sub-question error: {e}")
        raw_text = ""

    answer = parse_yes_no(raw_text)
    return answer, raw_text


# ---------------------------------------------------------------------------
# Decompose strategy: prompt templates (data-driven)
# ---------------------------------------------------------------------------

DEFAULT_SUB_PROMPTS = {
    'condition': (
        "Compare these two patients' conditions lists.\n\n"
        "Patient A:\n{section_a}\n\n"
        "Patient B:\n{section_b}\n\n"
        "Do these patients share any chronic or significant conditions? "
        "Answer YES or NO only."
    ),
    'medication': (
        "Compare these two patients' medication lists.\n\n"
        "Patient A:\n{section_a}\n\n"
        "Patient B:\n{section_b}\n\n"
        "Are any medications shared or treating the same conditions? "
        "Answer YES or NO only."
    ),
    'timeline': (
        "Compare these two patients' encounter timelines.\n\n"
        "Patient A:\n{section_a}\n\n"
        "Patient B:\n{section_b}\n\n"
        "Are these timelines compatible with one person visiting two different "
        "facilities? Answer YES or NO only."
    ),
    'contradiction': (
        "Compare these two patients' complete medical histories.\n\n"
        "Patient A:\n{section_a}\n\n"
        "Patient B:\n{section_b}\n\n"
        "Is there anything in these records that would be impossible or "
        "contradictory for a single person? Answer YES or NO only."
    ),
}

# Maps each sub-question to (section_key, fallback_text)
SUB_PROMPT_SECTIONS = {
    'condition': ('conditions', 'CONDITIONS: none'),
    'medication': ('medications', 'MEDICATIONS: none'),
    'timeline': ('encounters', 'ENCOUNTERS: none'),
    'contradiction': ('full', ''),
}


# Keep old builders for backward compatibility with any external callers
def build_condition_prompt(cond_a: str, cond_b: str) -> str:
    return DEFAULT_SUB_PROMPTS['condition'].format(section_a=cond_a, section_b=cond_b)

def build_medication_prompt(med_a: str, med_b: str) -> str:
    return DEFAULT_SUB_PROMPTS['medication'].format(section_a=med_a, section_b=med_b)

def build_timeline_prompt(enc_a: str, enc_b: str) -> str:
    return DEFAULT_SUB_PROMPTS['timeline'].format(section_a=enc_a, section_b=enc_b)

def build_contradiction_prompt(full_a: str, full_b: str) -> str:
    return DEFAULT_SUB_PROMPTS['contradiction'].format(section_a=full_a, section_b=full_b)


# ---------------------------------------------------------------------------
# Decompose strategy: voting + evaluation
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    'condition': 2,
    'medication': 1,
    'timeline': 1,
    'contradiction': 2,
}


def aggregate_votes(condition: bool | None, medication: bool | None,
                    timeline: bool | None, contradiction: bool | None,
                    weights: dict | None = None,
                    threshold: int = 3) -> tuple[bool, float]:
    """
    Weighted voting over sub-question answers.

    Returns (is_match, confidence) where confidence = score / max_score.
    None answers contribute 0 (conservative).
    """
    w = weights or _DEFAULT_WEIGHTS

    score = 0
    max_score = w['condition'] + w['medication'] + w['timeline'] + w['contradiction']

    # Positive signals: yes → add weight
    if condition is True:
        score += w['condition']
    if medication is True:
        score += w['medication']
    if timeline is True:
        score += w['timeline']

    # Reversed polarity: no contradiction → add weight
    if contradiction is False:
        score += w['contradiction']

    confidence = score / max_score if max_score > 0 else 0.0
    is_match = score >= threshold

    return is_match, confidence


def evaluate_decomposed(examples: list[dict], ollama_client: OpenAI,
                        model_name: str, temperature: float,
                        max_tokens_sub: int, weights: dict | None,
                        threshold: int,
                        sub_prompts: dict | None = None) -> dict:
    """
    Evaluate examples using the decompose strategy.

    For each pair: extract sections → 4 sub-question calls → aggregate.
    Returns same dict format as evaluate_prompt() plus sub-question details.

    Args:
        sub_prompts: Optional dict of prompt templates with {section_a}/{section_b}
                     placeholders. Defaults to DEFAULT_SUB_PROMPTS.
    """
    prompts = sub_prompts or DEFAULT_SUB_PROMPTS
    sub_keys = ['condition', 'medication', 'timeline', 'contradiction']

    results = []
    details = []
    tp = fp = tn = fn = 0

    for i, ex in enumerate(examples):
        sec_a = extract_sections(ex['history_a'])
        sec_b = extract_sections(ex['history_b'])

        # Call all sub-questions via loop over SUB_PROMPT_SECTIONS
        answers = {}
        raws = {}
        for key in sub_keys:
            section_key, fallback = SUB_PROMPT_SECTIONS[key]
            text_a = sec_a.get(section_key, fallback)
            text_b = sec_b.get(section_key, fallback)
            prompt = prompts[key].format(section_a=text_a, section_b=text_b)
            ans, raw = call_sub_question(
                prompt, ollama_client, model_name, temperature, max_tokens_sub,
            )
            answers[key] = ans
            raws[key] = raw

        # Aggregate
        is_match, confidence = aggregate_votes(
            answers['condition'], answers['medication'],
            answers['timeline'], answers['contradiction'],
            weights=weights, threshold=threshold,
        )

        # Log first few examples for debugging
        if i < 3:
            logger.info(
                f"  [{i}] actual={ex['is_match']} predicted={is_match} "
                f"(cond={answers['condition']} med={answers['medication']} "
                f"time={answers['timeline']} contra={answers['contradiction']} "
                f"conf={confidence:.2f})"
            )

        # Confusion matrix
        actual = ex['is_match']
        if is_match and actual:
            tp += 1
        elif is_match and not actual:
            fp += 1
        elif not is_match and actual:
            fn += 1
        else:
            tn += 1

        results.append({
            'index': i,
            'predicted': is_match,
            'actual': actual,
            'confidence': confidence,
            'correct': is_match == actual,
        })

        details.append({
            'index': i,
            'record_id_1': ex.get('record_id_1', ''),
            'record_id_2': ex.get('record_id_2', ''),
            'actual': actual,
            'predicted': is_match,
            'confidence': confidence,
            'sub_questions': {
                key: {'answer': answers[key], 'raw': raws[key][:200]}
                for key in sub_keys
            },
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(examples):
            logger.info(f"  Evaluated {i + 1}/{len(examples)}")

    total = len(examples)
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        'results': results,
        'details': details,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'parse_failures': 0,
        'total': total,
    }


async def run_decompose_evaluation(config: dict,
                                   experiment_name: str | None = None,
                                   threshold: int | None = None):
    """
    Run the decompose strategy: one evaluation pass with sub-question voting.

    No Claude optimization loop — just section decomposition + weighted voting.
    """
    # Determine experiment output directory
    if experiment_name is None:
        experiment_name = datetime.now().strftime("decompose_%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "output" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update latest symlink
    latest_link = Path(__file__).parent / "output" / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(experiment_name)

    logger.info(f"Experiment: {experiment_name} (decompose strategy)")
    logger.info(f"Output dir: {output_dir.resolve()}")

    # Load decompose config
    decompose_config = config.get('decompose', {})
    weights = decompose_config.get('weights', _DEFAULT_WEIGHTS)
    max_tokens_sub = decompose_config.get('max_tokens_sub', 64)
    if threshold is None:
        threshold = decompose_config.get('threshold', 3)

    # Save metadata
    _save_experiment_metadata(output_dir, config, 0, experiment_name,
                              strategy='decompose',
                              decompose_config={
                                  'threshold': threshold,
                                  'max_tokens_sub': max_tokens_sub,
                                  'weights': weights,
                              })

    # Load training data
    logger.info("Loading training data...")
    all_examples = load_training_examples(config)

    if len(all_examples) < 4:
        logger.error(f"Only {len(all_examples)} examples — need at least 4")
        return

    # Split train/val (75/25) — same as DSPy pipeline
    split_idx = int(len(all_examples) * 0.75)
    trainset = all_examples[:split_idx]
    valset = all_examples[split_idx:]
    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Setup MedGemma client
    model_config = config['model']
    ollama_client = OpenAI(
        base_url=model_config['api_base'],
        api_key=model_config['api_key'],
    )
    model_name = model_config['name']
    temperature = model_config.get('temperature', 0.1)

    logger.info(f"Decompose config: threshold={threshold}, weights={weights}, "
                f"max_tokens_sub={max_tokens_sub}")

    # Evaluate on training set
    logger.info("Evaluating on training set...")
    train_start = time.time()
    train_result = evaluate_decomposed(
        trainset, ollama_client, model_name, temperature,
        max_tokens_sub, weights, threshold,
    )
    train_elapsed = time.time() - train_start
    logger.info(f"  Train accuracy: {train_result['accuracy']:.3f} "
                f"({train_elapsed:.0f}s)")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_start = time.time()
    val_result = evaluate_decomposed(
        valset, ollama_client, model_name, temperature,
        max_tokens_sub, weights, threshold,
    )
    val_elapsed = time.time() - val_start
    logger.info(f"  Val accuracy: {val_result['accuracy']:.3f} "
                f"({val_elapsed:.0f}s)")

    # Save decompose details
    all_details = {
        'train': train_result['details'],
        'val': val_result['details'],
    }
    with open(output_dir / "decompose_details.json", "w") as f:
        json.dump(all_details, f, indent=2, default=str)

    # Build comparison report (compatible format)
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'decompose',
        'decompose_config': {
            'threshold': threshold,
            'max_tokens_sub': max_tokens_sub,
            'weights': weights,
        },
        'train': {
            'accuracy': train_result['accuracy'],
            'tp': train_result['tp'],
            'fp': train_result['fp'],
            'tn': train_result['tn'],
            'fn': train_result['fn'],
            'elapsed_seconds': round(train_elapsed, 1),
        },
        'val': {
            'accuracy': val_result['accuracy'],
            'tp': val_result['tp'],
            'fp': val_result['fp'],
            'tn': val_result['tn'],
            'fn': val_result['fn'],
            'elapsed_seconds': round(val_elapsed, 1),
        },
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("DECOMPOSE EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Strategy: decompose (threshold={threshold}, weights={weights})")
    print(f"\n{'Split':<10} {'Acc':>7} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'Time':>8}")
    print("-" * 45)
    print(f"{'Train':<10} {train_result['accuracy']:>7.3f} "
          f"{train_result['tp']:>4} {train_result['fp']:>4} "
          f"{train_result['tn']:>4} {train_result['fn']:>4} "
          f"{train_elapsed:>7.0f}s")
    print(f"{'Val':<10} {val_result['accuracy']:>7.3f} "
          f"{val_result['tp']:>4} {val_result['fp']:>4} "
          f"{val_result['tn']:>4} {val_result['fn']:>4} "
          f"{val_elapsed:>7.0f}s")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir.resolve()}")


async def run_decompose_optimization(config: dict, max_iterations: int,
                                      experiment_name: str | None = None,
                                      threshold: int | None = None,
                                      initial_prompts_path: Path | None = None):
    """
    Run the Claude-powered decompose sub-prompt optimization loop.

    Iteratively evaluates, analyzes errors, and asks Claude to improve
    the 4 sub-question templates, weights, and threshold.
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("decompose-opt_%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "output" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update latest symlink
    latest_link = Path(__file__).parent / "output" / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(experiment_name)

    logger.info(f"Experiment: {experiment_name} (decompose optimization)")
    logger.info(f"Output dir: {output_dir.resolve()}")

    # Load decompose config
    decompose_config = config.get('decompose', {})
    weights = decompose_config.get('weights', _DEFAULT_WEIGHTS)
    max_tokens_sub = decompose_config.get('max_tokens_sub', 64)
    if threshold is None:
        threshold = decompose_config.get('threshold', 3)

    claude_model = config.get('claude', {}).get('model', 'claude-opus-4-6')

    # Save metadata
    decompose_meta = {
        'threshold': threshold,
        'max_tokens_sub': max_tokens_sub,
        'weights': weights,
    }
    if initial_prompts_path:
        decompose_meta['warm_start_from'] = str(initial_prompts_path)
    _save_experiment_metadata(output_dir, config, max_iterations, experiment_name,
                              strategy='decompose-opt',
                              decompose_config=decompose_meta)

    # Load training data
    logger.info("Loading training data...")
    all_examples = load_training_examples(config)

    if len(all_examples) < 4:
        logger.error(f"Only {len(all_examples)} examples — need at least 4")
        return

    # Split train/val (75/25)
    split_idx = int(len(all_examples) * 0.75)
    trainset = all_examples[:split_idx]
    valset = all_examples[split_idx:]
    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Setup MedGemma client
    model_config = config['model']
    ollama_client = OpenAI(
        base_url=model_config['api_base'],
        api_key=model_config['api_key'],
    )
    model_name = model_config['name']
    temperature = model_config.get('temperature', 0.1)

    # Initialize state — warm-start from previous run if provided
    if initial_prompts_path:
        logger.info(f"Warm-starting from {initial_prompts_path}")
        with open(initial_prompts_path) as f:
            prev = json.load(f)
        current_prompts = prev['prompts']
        current_weights = prev.get('weights', dict(weights))
        current_threshold = prev.get('threshold', threshold)
        logger.info(f"  Loaded prompts from iteration {prev.get('iteration', '?')} "
                    f"(val_accuracy={prev.get('val_accuracy', '?')})")
    else:
        current_prompts = dict(DEFAULT_SUB_PROMPTS)
        current_weights = dict(weights)
        current_threshold = threshold

    iteration_history = []
    best_val_accuracy = -1.0
    best_iteration = 0
    best_prompts = dict(current_prompts)
    best_weights = dict(current_weights)
    best_threshold = current_threshold
    best_details = None

    for iteration in range(max_iterations + 1):
        iter_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"DECOMPOSE OPT — ITERATION {iteration}" +
                    (" (baseline)" if iteration == 0 else ""))
        logger.info(f"{'='*60}")
        logger.info(f"  Weights: {current_weights}, Threshold: {current_threshold}")

        # Evaluate on training set
        logger.info("Evaluating on training set...")
        train_result = evaluate_decomposed(
            trainset, ollama_client, model_name, temperature,
            max_tokens_sub, current_weights, current_threshold,
            sub_prompts=current_prompts,
        )
        train_elapsed = time.time() - iter_start
        logger.info(f"  Train accuracy: {train_result['accuracy']:.3f} "
                    f"({train_elapsed:.0f}s)")

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_start = time.time()
        val_result = evaluate_decomposed(
            valset, ollama_client, model_name, temperature,
            max_tokens_sub, current_weights, current_threshold,
            sub_prompts=current_prompts,
        )
        val_elapsed = time.time() - val_start
        total_elapsed = time.time() - iter_start
        logger.info(f"  Val accuracy: {val_result['accuracy']:.3f} "
                    f"({val_elapsed:.0f}s)")

        # Record iteration
        record = {
            'iteration': iteration,
            'train_accuracy': train_result['accuracy'],
            'val_accuracy': val_result['accuracy'],
            'tp': val_result['tp'],
            'fp': val_result['fp'],
            'tn': val_result['tn'],
            'fn': val_result['fn'],
            'train_tp': train_result['tp'],
            'train_fp': train_result['fp'],
            'train_tn': train_result['tn'],
            'train_fn': train_result['fn'],
            'weights': dict(current_weights),
            'threshold': current_threshold,
            'prompts': dict(current_prompts),
            'elapsed_seconds': round(total_elapsed, 1),
        }
        iteration_history.append(record)

        # Track best by val accuracy
        if val_result['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_result['accuracy']
            best_iteration = iteration
            best_prompts = dict(current_prompts)
            best_weights = dict(current_weights)
            best_threshold = current_threshold
            best_details = {
                'train': train_result['details'],
                'val': val_result['details'],
            }
            logger.info(f"  New best! Val accuracy: {best_val_accuracy:.3f}")

        # Print summary
        is_best = " *** BEST ***" if iteration == best_iteration else ""
        print(f"\n  Iteration {iteration}: "
              f"train={train_result['accuracy']:.3f} "
              f"val={val_result['accuracy']:.3f}{is_best}")

        # Stop if this was the last iteration
        if iteration >= max_iterations:
            break

        # Build error analysis and call Claude
        logger.info("Building error analysis...")
        error_analysis = build_decompose_error_analysis(
            train_result, trainset, current_prompts,
        )

        try:
            new_prompts, new_weights, new_threshold, analysis = \
                await generate_improved_sub_prompts(
                    current_prompts, error_analysis,
                    iteration_history, trainset, claude_model,
                )
            logger.info(f"Claude analysis: {analysis[:300]}...")
            record['claude_analysis'] = analysis
            current_prompts = new_prompts
            current_weights = new_weights
            current_threshold = new_threshold
        except Exception as e:
            logger.error(f"Claude call failed: {e}")
            logger.info("Keeping current prompts for next iteration")

    # Save outputs
    # 1. Optimization history (strip full prompts to keep file manageable)
    serializable_history = []
    for h in iteration_history:
        sh = {k: v for k, v in h.items() if k != 'prompts'}
        sh['prompt_lengths'] = {k: len(v) for k, v in h['prompts'].items()}
        serializable_history.append(sh)
    with open(output_dir / "optimization_history.json", "w") as f:
        json.dump(serializable_history, f, indent=2, default=str)

    # 2. Best sub-prompts
    best_sub_prompts = {
        'iteration': best_iteration,
        'val_accuracy': best_val_accuracy,
        'prompts': best_prompts,
        'weights': best_weights,
        'threshold': best_threshold,
    }
    with open(output_dir / "best_sub_prompts.json", "w") as f:
        json.dump(best_sub_prompts, f, indent=2)

    # 3. Decompose details for best iteration
    if best_details:
        with open(output_dir / "decompose_details.json", "w") as f:
            json.dump(best_details, f, indent=2, default=str)

    # 4. Comparison report
    baseline = iteration_history[0] if iteration_history else {}
    best_record = iteration_history[best_iteration] if best_iteration < len(iteration_history) else {}
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'decompose-opt',
        'total_iterations': max_iterations,
        'best_iteration': best_iteration,
        'baseline': {
            'train_accuracy': baseline.get('train_accuracy', 0),
            'val_accuracy': baseline.get('val_accuracy', 0),
        },
        'best': {
            'train_accuracy': best_record.get('train_accuracy', 0),
            'val_accuracy': best_record.get('val_accuracy', 0),
            'weights': best_weights,
            'threshold': best_threshold,
        },
        'train': {
            'accuracy': best_record.get('train_accuracy', 0),
            'tp': best_record.get('train_tp', 0),
            'fp': best_record.get('train_fp', 0),
            'tn': best_record.get('train_tn', 0),
            'fn': best_record.get('train_fn', 0),
        },
        'val': {
            'accuracy': best_record.get('val_accuracy', 0),
            'tp': best_record.get('tp', 0),
            'fp': best_record.get('fp', 0),
            'tn': best_record.get('tn', 0),
            'fn': best_record.get('fn', 0),
        },
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print final report
    print("\n" + "=" * 70)
    print("DECOMPOSE OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best iteration: {best_iteration} / {max_iterations}")
    print(f"\n{'Iteration':<12} {'Train':>7} {'Val':>7}")
    print("-" * 30)
    print(f"{'Baseline':<12} {baseline.get('train_accuracy', 0):>7.3f} "
          f"{baseline.get('val_accuracy', 0):>7.3f}")
    print(f"{'Best (#' + str(best_iteration) + ')':<12} "
          f"{best_record.get('train_accuracy', 0):>7.3f} "
          f"{best_record.get('val_accuracy', 0):>7.3f}")
    print(f"\nBest weights: {best_weights}")
    print(f"Best threshold: {best_threshold}")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompt(instruction: str, few_shots: list[dict],
                    history_a: str, history_b: str,
                    max_few_shots: int = 2,
                    few_shot_truncate: int = 200) -> str:
    """Build the full prompt string from components.

    Keeps few-shots short to stay within MedGemma 4B's context budget.
    """
    parts = [instruction.strip()]

    # Cap few-shots to avoid filling context window
    shots_to_use = few_shots[:max_few_shots] if few_shots else []
    if shots_to_use:
        parts.append("\n--- EXAMPLES ---")
        for i, shot in enumerate(shots_to_use, 1):
            parts.append(f"\nExample {i}:")
            parts.append(f"Patient A:\n{shot['history_a'][:few_shot_truncate]}")
            parts.append(f"Patient B:\n{shot['history_b'][:few_shot_truncate]}")
            parts.append(f"Answer: is_match: {str(shot['is_match']).lower()}, "
                         f"confidence: {shot.get('confidence', '0.90')}")
        parts.append("\n--- END EXAMPLES ---\n")

    parts.append("Now compare these two patients:")
    parts.append(f"\nPatient A medical history:\n{history_a}")
    parts.append(f"\nPatient B medical history:\n{history_b}")
    parts.append("\nRespond with is_match and confidence only.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MedGemma evaluation
# ---------------------------------------------------------------------------

def evaluate_prompt(instruction: str, few_shots: list[dict],
                    examples: list[dict], ollama_client: OpenAI,
                    model_name: str, temperature: float,
                    max_tokens: int, confidence_threshold: float) -> dict:
    """
    Evaluate a prompt against all examples using MedGemma.

    Returns dict with per-example results and aggregate stats.
    """
    results = []
    tp = fp = tn = fn = parse_failures = 0

    for i, ex in enumerate(examples):
        prompt = assemble_prompt(instruction, few_shots,
                                 ex['history_a'], ex['history_b'])

        # Log prompt length for first example to track context budget
        if i == 0:
            logger.info(f"  Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

        try:
            response = ollama_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"MedGemma API error on example {i}: {e}")
            raw_text = ""

        # Log first 3 raw responses for debugging
        if i < 3:
            cleaned = extract_answer(raw_text)
            logger.info(f"  Raw response [{i}] (actual={ex['is_match']}): "
                         f"{cleaned[:150]}")

        is_match, confidence, parse_error = parse_medgemma_response(raw_text)

        # Apply confidence threshold
        if is_match and confidence is not None and confidence < confidence_threshold:
            is_match = False

        # Default to False on parse failure
        if is_match is None:
            is_match = False
            parse_failures += 1

        # Confusion matrix
        actual = ex['is_match']
        if is_match and actual:
            tp += 1
        elif is_match and not actual:
            fp += 1
        elif not is_match and actual:
            fn += 1
        else:
            tn += 1

        results.append({
            'index': i,
            'predicted': is_match,
            'actual': actual,
            'confidence': confidence,
            'parse_error': parse_error,
            'raw_response': raw_text[:300],
            'correct': is_match == actual,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            logger.info(f"  Evaluated {i + 1}/{len(examples)}")

    total = len(examples)
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        'results': results,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'parse_failures': parse_failures,
        'total': total,
    }


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def build_error_analysis(eval_result: dict, examples: list[dict]) -> str:
    """Generate a structured error report for Claude to analyze."""
    lines = []
    lines.append("=== EVALUATION RESULTS ===")
    lines.append(f"Accuracy: {eval_result['accuracy']:.3f} "
                 f"({eval_result['tp'] + eval_result['tn']}/{eval_result['total']})")
    lines.append(f"Confusion matrix: TP={eval_result['tp']} FP={eval_result['fp']} "
                 f"TN={eval_result['tn']} FN={eval_result['fn']}")
    lines.append(f"Parse failures: {eval_result['parse_failures']}/{eval_result['total']}")

    # Dataset stats
    n_match = sum(1 for e in examples if e['is_match'])
    n_nonmatch = len(examples) - n_match
    lines.append(f"Dataset: {n_match} true matches, {n_nonmatch} non-matches "
                 f"(base rate: {n_match/len(examples):.0%} matches)")

    # Show sample raw responses (both correct and incorrect)
    lines.append("\n=== SAMPLE RAW MODEL RESPONSES ===")
    for r in eval_result['results'][:5]:
        idx = r['index']
        tag = "CORRECT" if r['correct'] else "WRONG"
        lines.append(f"[{tag}] Example {idx} (actual={r['actual']}): {r['raw_response'][:200]}")

    # Detailed failing examples (up to 5)
    failures = [r for r in eval_result['results'] if not r['correct']]
    lines.append(f"\n=== FAILING EXAMPLES ({len(failures)} total, showing up to 5) ===")

    for fail in failures[:5]:
        idx = fail['index']
        ex = examples[idx]
        lines.append(f"\n--- Example {idx} ---")
        lines.append(f"Predicted: {fail['predicted']}, Actual: {fail['actual']}")
        lines.append(f"Confidence: {fail['confidence']}")
        if fail['parse_error']:
            lines.append(f"Parse error: {fail['parse_error']}")
        lines.append(f"Raw response (truncated): {fail['raw_response'][:200]}")
        # Truncate histories to keep meta-prompt manageable
        lines.append(f"Patient A history (truncated):\n{ex['history_a'][:400]}")
        lines.append(f"Patient B history (truncated):\n{ex['history_b'][:400]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decompose error analysis + Claude optimization
# ---------------------------------------------------------------------------

def build_decompose_error_analysis(train_result: dict,
                                    examples: list[dict],
                                    current_prompts: dict) -> str:
    """
    Build a structured error report for decompose sub-question optimization.

    Includes per-sub-question answer distributions for matches vs non-matches,
    sample error pairs with raw responses, and current prompt templates.
    """
    lines = []
    sub_keys = ['condition', 'medication', 'timeline', 'contradiction']
    details = train_result['details']

    lines.append("=== OVERALL RESULTS ===")
    lines.append(f"Accuracy: {train_result['accuracy']:.3f} "
                 f"({train_result['tp'] + train_result['tn']}/{train_result['total']})")
    lines.append(f"Confusion matrix: TP={train_result['tp']} FP={train_result['fp']} "
                 f"TN={train_result['tn']} FN={train_result['fn']}")

    n_match = sum(1 for e in examples if e['is_match'])
    n_nonmatch = len(examples) - n_match
    lines.append(f"Dataset: {n_match} matches, {n_nonmatch} non-matches")

    # Per-sub-question statistics
    lines.append("\n=== PER-SUB-QUESTION STATISTICS ===")
    for key in sub_keys:
        lines.append(f"\n--- {key.upper()} ---")

        # Collect answers grouped by actual label
        match_answers = {'True': 0, 'False': 0, 'None': 0}
        nonmatch_answers = {'True': 0, 'False': 0, 'None': 0}

        for d, ex in zip(details, examples):
            ans = d['sub_questions'][key]['answer']
            ans_str = str(ans)
            if ex['is_match']:
                match_answers[ans_str] = match_answers.get(ans_str, 0) + 1
            else:
                nonmatch_answers[ans_str] = nonmatch_answers.get(ans_str, 0) + 1

        lines.append(f"  Actual MATCHES   → YES: {match_answers['True']}, "
                     f"NO: {match_answers['False']}, UNPARSED: {match_answers['None']}")
        lines.append(f"  Actual NON-MATCH → YES: {nonmatch_answers['True']}, "
                     f"NO: {nonmatch_answers['False']}, UNPARSED: {nonmatch_answers['None']}")

        # Individual accuracy for this sub-question
        if key == 'contradiction':
            # Reversed polarity: YES=contradiction → non-match, NO → match
            correct = sum(1 for d, ex in zip(details, examples)
                         if (d['sub_questions'][key]['answer'] is False) == ex['is_match'])
        else:
            # Normal: YES → match, NO → non-match
            correct = sum(1 for d, ex in zip(details, examples)
                         if d['sub_questions'][key]['answer'] is not None
                         and (d['sub_questions'][key]['answer'] == ex['is_match']))
        total_parseable = sum(1 for d in details
                              if d['sub_questions'][key]['answer'] is not None)
        if total_parseable > 0:
            lines.append(f"  Individual accuracy: {correct}/{total_parseable} "
                         f"= {correct/total_parseable:.1%}")
        else:
            lines.append("  Individual accuracy: N/A (all unparsed)")

    # Sample error pairs with all sub-question responses
    failures = [(d, examples[d['index']]) for d in details
                if d['predicted'] != d['actual']]
    lines.append(f"\n=== SAMPLE ERRORS ({len(failures)} total, showing up to 5) ===")
    for d, ex in failures[:5]:
        lines.append(f"\n--- Pair: {d.get('record_id_1', '?')} vs {d.get('record_id_2', '?')} ---")
        lines.append(f"Actual: {'MATCH' if d['actual'] else 'NON-MATCH'}, "
                     f"Predicted: {'MATCH' if d['predicted'] else 'NON-MATCH'}")
        for key in sub_keys:
            sq = d['sub_questions'][key]
            lines.append(f"  {key}: answer={sq['answer']}, raw={sq['raw'][:150]}")

    # Current prompt templates
    lines.append("\n=== CURRENT PROMPT TEMPLATES ===")
    for key in sub_keys:
        lines.append(f"\n--- {key.upper()} ---")
        lines.append(current_prompts[key])

    return "\n".join(lines)


async def generate_improved_sub_prompts(current_prompts: dict,
                                         error_analysis: str,
                                         iteration_history: list[dict],
                                         examples: list[dict],
                                         claude_model: str) -> tuple:
    """
    Ask Claude to improve the 4 decompose sub-question prompts.

    Returns:
        (new_prompts: dict, new_weights: dict, new_threshold: int, analysis: str)
    """
    # Build iteration history summary
    history_summary = ""
    if iteration_history:
        history_lines = []
        for h in iteration_history:
            history_lines.append(
                f"  Iteration {h['iteration']}: train_acc={h['train_accuracy']:.3f}, "
                f"val_acc={h['val_accuracy']:.3f} "
                f"(TP={h['tp']} FP={h['fp']} TN={h['tn']} FN={h['fn']})"
            )
        history_summary = "Previous iterations:\n" + "\n".join(history_lines)

    n_match = sum(1 for e in examples if e['is_match'])

    meta_prompt = f"""You are an expert prompt engineer optimizing sub-question prompts for a small medical AI model (MedGemma 4B, running locally via Ollama).

TASK: We use a "decompose" strategy for medical record entity resolution. Instead of one big prompt, we ask 4 focused sub-questions about pairs of patient records from different hospitals, then aggregate votes to decide if the records belong to the same person.

The 4 sub-questions are:
1. CONDITION: Do the patients share chronic/significant conditions? (YES = evidence for match)
2. MEDICATION: Do the patients share medications? (YES = evidence for match)
3. TIMELINE: Are encounter timelines compatible with one person at two facilities? (YES = evidence for match)
4. CONTRADICTION: Is there anything contradictory for a single person? (YES = evidence AGAINST match — REVERSED POLARITY)

IMPORTANT MODEL BEHAVIOR:
- MedGemma 4B outputs thinking tokens before the answer: <unused94>thought...<unused95> then the actual answer
- We parse YES/NO from the text AFTER stripping thinking tokens
- The model has ~8K token context window — keep prompts concise
- The model tends to be biased (often says YES to everything, or NO to everything)
- Your job: craft prompts that produce DISCRIMINATIVE answers — different for true matches vs non-matches
- Dataset has {n_match}/{len(examples)} matches ({n_match/len(examples):.0%} match rate)

VOTING SYSTEM:
- Each sub-question contributes a weighted vote (condition, medication, timeline positive; contradiction reversed)
- Weighted sum >= threshold → predict MATCH
- You can suggest new weights and threshold

{history_summary}

{error_analysis}

INSTRUCTIONS:
1. Analyze which sub-questions are broken (non-discriminative) and why
2. Rewrite prompts to elicit more discriminative YES/NO answers from MedGemma 4B
3. Each prompt MUST contain {{section_a}} and {{section_b}} placeholders (these get filled with the relevant patient sections)
4. Each prompt MUST end with "Answer YES or NO only."
5. Keep prompts concise (under 150 words each)
6. Consider: adding calibration context, reframing questions, using negative examples, emphasizing what "same patient" means

YOUR RESPONSE MUST USE THESE EXACT XML TAGS:

<analysis>
Your analysis of what's going wrong and strategy for each sub-question.
</analysis>

<condition_prompt>
The new condition comparison prompt template (must include {{section_a}} and {{section_b}}).
</condition_prompt>

<medication_prompt>
The new medication comparison prompt template.
</medication_prompt>

<timeline_prompt>
The new timeline comparison prompt template.
</timeline_prompt>

<contradiction_prompt>
The new contradiction detection prompt template.
</contradiction_prompt>

<weights>
JSON object with integer weights, e.g. {{"condition": 2, "medication": 1, "timeline": 1, "contradiction": 2}}
</weights>

<threshold>
Integer threshold for weighted voting (e.g. 3)
</threshold>"""

    logger.info("Calling Claude for sub-prompt improvement...")
    response = await call_claude(meta_prompt, model=claude_model)
    logger.info(f"Claude response length: {len(response)} chars")

    # Parse response
    new_prompts = dict(current_prompts)  # fallback to current

    for key in ['condition', 'medication', 'timeline', 'contradiction']:
        tag = f'{key}_prompt'
        match = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # Validate placeholders exist
            if '{section_a}' in candidate and '{section_b}' in candidate:
                new_prompts[key] = candidate
                logger.info(f"  Extracted new {key} prompt")
            else:
                logger.warning(f"  {key} prompt missing placeholders, keeping current")
        else:
            logger.warning(f"  Could not extract {key} prompt, keeping current")

    # Parse weights
    new_weights = _DEFAULT_WEIGHTS.copy()
    weights_match = re.search(r'<weights>(.*?)</weights>', response, re.DOTALL)
    if weights_match:
        try:
            parsed = json.loads(weights_match.group(1).strip())
            if all(k in parsed for k in _DEFAULT_WEIGHTS):
                new_weights = {k: int(v) for k, v in parsed.items()
                               if k in _DEFAULT_WEIGHTS}
                logger.info(f"  New weights: {new_weights}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"  Could not parse weights: {e}")

    # Parse threshold
    new_threshold = 3
    threshold_match = re.search(r'<threshold>(.*?)</threshold>', response, re.DOTALL)
    if threshold_match:
        try:
            new_threshold = int(threshold_match.group(1).strip())
            logger.info(f"  New threshold: {new_threshold}")
        except ValueError as e:
            logger.warning(f"  Could not parse threshold: {e}")

    # Extract analysis
    analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else "No analysis extracted"

    return new_prompts, new_weights, new_threshold, analysis


# ---------------------------------------------------------------------------
# Claude interaction
# ---------------------------------------------------------------------------

async def call_claude(prompt: str, model: str = "claude-opus-4-6") -> str:
    """
    Call Claude via the claude-agent-sdk.

    Uses subscription auth (no API key needed). The SDK shells out to the
    local `claude` CLI.
    """
    from claude_agent_sdk import query, ClaudeAgentOptions

    result_parts = []
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model=model,
            max_turns=1,
            allowed_tools=[],
            system_prompt=(
                "You are a prompt engineering expert. "
                "Always respond using the exact XML tag format requested in the prompt. "
                "Do not use any tools. Respond with text only."
            ),
        ),
    ):
        # AssistantMessage has .content list of blocks
        if hasattr(message, 'content'):
            if isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, 'text'):
                        result_parts.append(block.text)
            elif isinstance(message.content, str):
                result_parts.append(message.content)

    return "\n".join(result_parts)


async def generate_improved_prompt(current_instruction: str,
                                   current_few_shots: list[dict],
                                   error_analysis: str,
                                   iteration_history: list[dict],
                                   all_examples: list[dict],
                                   claude_model: str) -> tuple:
    """
    Ask Claude to analyze errors and generate an improved prompt.

    Returns:
        (new_instruction: str, new_few_shots: list[dict])
    """
    # Build iteration history summary
    history_summary = ""
    if iteration_history:
        history_lines = []
        for h in iteration_history:
            history_lines.append(
                f"  Iteration {h['iteration']}: accuracy={h['accuracy']:.3f} "
                f"(TP={h['tp']} FP={h['fp']} TN={h['tn']} FN={h['fn']}, "
                f"parse_fail={h['parse_failures']})"
            )
        history_summary = "Previous iterations:\n" + "\n".join(history_lines)

    # Build few-shot candidate pool description
    match_examples = [e for e in all_examples if e['is_match']]
    non_match_examples = [e for e in all_examples if not e['is_match']]
    n_match = len(match_examples)

    # Build few-shot summary outside f-string to avoid brace escaping issues
    if current_few_shots:
        fs_summary_items = [
            {"is_match": s.get('is_match'),
             "history_a_len": len(s.get('history_a', '')),
             "history_b_len": len(s.get('history_b', ''))}
            for s in current_few_shots
        ]
        fs_summary = json.dumps(fs_summary_items, indent=2)
    else:
        fs_summary = "None (zero-shot)"

    meta_prompt = f"""You are an expert prompt engineer optimizing a prompt for a small medical AI model (MedGemma 4B, running locally via Ollama).

TASK: The model compares two patients' medical histories (from different hospitals) and determines if they are the same person. It outputs is_match (true/false) and confidence (0.0-1.0).

CURRENT PROMPT:
<current_prompt>
{current_instruction}
</current_prompt>

CURRENT FEW-SHOT EXAMPLES:
{fs_summary}

{history_summary}

{error_analysis}

IMPORTANT CONSTRAINTS:
- MedGemma 4B is a VERY small model (~4B params) with ~8K token context window
- CONTEXT BUDGET IS CRITICAL: Each patient history is ~1000-3000 chars. Few-shot examples are truncated to 200 chars each. Total prompt must fit in ~6000 tokens to leave room for the actual patient histories.
- Keep the instruction concise — under 300 words, ideally under 200
- The model must output EXACTLY: is_match: true/false and confidence: 0.XX
- A confidence threshold is applied: predictions with confidence below threshold are converted to false. Current threshold: 0.5
- The model only sees medical records, NOT demographics (no names, DOB, SSN, etc.)
- These are synthetic Synthea records from different hospitals for the SAME patients. Records at different facilities will look different — partial overlap is normal and expected.
- The dataset is {n_match}/{len(all_examples)} matches ({n_match/len(all_examples):.0%} match rate)

AVAILABLE EXAMPLES FOR FEW-SHOTS:
- {len(match_examples)} true match pairs available (indices: {[i for i,e in enumerate(all_examples) if e['is_match']][:10]}...)
- {len(non_match_examples)} non-match pairs available (indices: {[i for i,e in enumerate(all_examples) if not e['is_match']][:10]}...)
- You can select 0-2 examples by index (max 2 to save context budget)
- Choose examples that demonstrate the decision boundary — one clear match and one clear non-match work best

YOUR RESPONSE MUST USE THESE EXACT XML TAGS:

<analysis>
Your analysis of what's going wrong and your strategy for improvement.
</analysis>

<prompt>
The new instruction text to use (replaces the current prompt instruction).
</prompt>

<few_shots>
A JSON array of example indices to use as few-shots, e.g. [0, 5] or [] for zero-shot.
Each index refers to the training examples (0-indexed). Maximum 2.
</few_shots>"""

    logger.info("Calling Claude for prompt improvement...")
    response = await call_claude(meta_prompt, model=claude_model)
    logger.info(f"Claude response length: {len(response)} chars")

    # Parse structured response
    new_instruction = current_instruction  # fallback
    new_few_shots = current_few_shots  # fallback

    # Extract prompt — try XML tags first, then markdown code blocks as fallback
    prompt_match = re.search(r'<prompt>(.*?)</prompt>', response, re.DOTALL)
    if not prompt_match:
        # Fallback: look for ```prompt ... ``` or a section labeled "Prompt:"
        prompt_match = re.search(r'```prompt\s*\n(.*?)```', response, re.DOTALL)
    if not prompt_match:
        # Fallback: look for text between "PROMPT:" and the next section
        prompt_match = re.search(r'(?:NEW |IMPROVED )?PROMPT:\s*\n(.*?)(?:\n(?:FEW|ANALYSIS)|$)',
                                  response, re.DOTALL)
    if prompt_match:
        new_instruction = prompt_match.group(1).strip()
        logger.info("Extracted new instruction from Claude response")
    else:
        logger.warning("Could not extract prompt from Claude response, keeping current")
        logger.warning(f"Claude response (first 500 chars): {response[:500]}")

    # Extract few-shot indices — try XML tags first, then JSON array fallback
    fs_match = re.search(r'<few_shots>(.*?)</few_shots>', response, re.DOTALL)
    if not fs_match:
        fs_match = re.search(r'```few_shots\s*\n(.*?)```', response, re.DOTALL)
    if not fs_match:
        # Look for any JSON array in the response after "few" keyword
        fs_match = re.search(r'few.shots?.*?(\[[\d,\s]*\])', response, re.DOTALL | re.IGNORECASE)
    if fs_match:
        try:
            raw_fs = fs_match.group(1).strip()
            indices = json.loads(raw_fs)
            new_few_shots = []
            for idx in indices[:2]:  # Cap at 2 few-shots
                if 0 <= idx < len(all_examples):
                    ex = all_examples[idx]
                    new_few_shots.append({
                        'history_a': ex['history_a'],
                        'history_b': ex['history_b'],
                        'is_match': ex['is_match'],
                        'confidence': '0.95' if ex['is_match'] else '0.10',
                    })
            logger.info(f"Selected {len(new_few_shots)} few-shot examples")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Could not parse few_shots JSON: {e}")

    # Extract analysis for logging — try XML tags first, then fallback
    analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL)
    if not analysis_match:
        analysis_match = re.search(r'```analysis\s*\n(.*?)```', response, re.DOTALL)
    if not analysis_match:
        analysis_match = re.search(r'(?:ANALYSIS|Analysis):\s*\n(.*?)(?:\n(?:PROMPT|NEW)|$)',
                                    response, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else "No analysis extracted"

    return new_instruction, new_few_shots, analysis


# ---------------------------------------------------------------------------
# Training data loading (reuses llm-entity-resolution infrastructure)
# ---------------------------------------------------------------------------

def load_training_examples(config: dict) -> list[dict]:
    """
    Load gray zone pairs with ground truth labels as plain dicts.

    Mirrors build_training_data() from llm-entity-resolution/src/optimize.py
    but returns plain dicts instead of dspy.Example objects.
    """
    run_dir = get_run_directory(config['base_dir'], config['run_id'])

    # Load patient data
    patients_df = load_facility_patients(str(run_dir))
    patients_df['record_id'] = (
        patients_df['facility_id'] + '_' + patients_df['Id'].astype(str)
    )

    # Load ground truth
    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    # Build true matching pairs set
    true_pairs = set()
    for _, group in ground_truth_df.groupby('true_patient_id'):
        rids = group['record_id'].dropna().tolist()
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                true_pairs.add(tuple(sorted([rids[i], rids[j]])))

    # Load gray zone pairs
    gz_config = config.get('gray_zone', {})
    features_csv = gz_config.get('features_csv',
                                 'entity-resolution/output/predicted_matches.csv')
    features_path = Path(config['base_dir']) / features_csv

    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)

    lower = gz_config.get('lower_threshold', 4.0)
    upper = gz_config.get('upper_threshold', 6.0)

    gray_zone_df = features_df[
        (features_df['total_score'] >= lower) &
        (features_df['total_score'] < upper)
    ]
    logger.info(f"Found {len(gray_zone_df)} gray zone pairs (score {lower}-{upper})")

    # Load medical records
    logger.info("Loading medical records...")
    medical_records = load_medical_records(str(run_dir))

    # Build record_id → (patient_uuid, facility_id) mapping
    record_map = {}
    for _, row in patients_df.iterrows():
        record_map[row['record_id']] = (row['Id'], row['facility_id'])

    # Generate examples
    examples = []
    for _, row in gray_zone_df.iterrows():
        rid1, rid2 = row['record_id_1'], row['record_id_2']

        if rid1 not in record_map or rid2 not in record_map:
            continue

        uuid1, fac1 = record_map[rid1]
        uuid2, fac2 = record_map[rid2]

        summary_a = summarize_patient_records(uuid1, fac1, medical_records)
        summary_b = summarize_patient_records(uuid2, fac2, medical_records)

        pair = tuple(sorted([rid1, rid2]))
        is_match = pair in true_pairs

        examples.append({
            'history_a': summary_a,
            'history_b': summary_b,
            'is_match': is_match,
            'record_id_1': rid1,
            'record_id_2': rid2,
        })

    matches = sum(1 for e in examples if e['is_match'])
    logger.info(f"Built {len(examples)} examples "
                f"({matches} matches, {len(examples) - matches} non-matches)")

    return examples


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def _get_git_hash() -> str:
    """Get short git hash for experiment tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _save_experiment_metadata(output_dir: Path, config: dict,
                              max_iterations: int, name: str,
                              strategy: str = 'single-shot',
                              decompose_config: dict | None = None):
    """Save experiment metadata for reproducibility."""
    metadata = {
        "name": name,
        "strategy": strategy,
        "timestamp": datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "max_iterations": max_iterations,
        "model": config.get("model", {}),
        "claude": config.get("claude", {}),
        "optimization": config.get("optimization", {}),
        "gray_zone": config.get("gray_zone", {}),
    }
    if strategy == 'single-shot':
        metadata["baseline_prompt"] = (
            Path(__file__).parent / "prompts" / "baseline_v1.txt"
        ).read_text().strip()
    if decompose_config:
        metadata["decompose"] = decompose_config
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def list_experiments():
    """Print a comparison table of all past experiments."""
    output_root = Path(__file__).parent / "output"
    if not output_root.exists():
        print("No experiments found.")
        return

    experiments = []
    for d in sorted(output_root.iterdir()):
        if not d.is_dir() or d.name == "latest":
            continue
        comp_path = d / "comparison.json"
        meta_path = d / "metadata.json"
        if not comp_path.exists():
            continue

        with open(comp_path) as f:
            comp = json.load(f)
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        strategy = meta.get("strategy", comp.get("strategy", "single-shot"))

        if strategy == 'decompose':
            # Decompose format: train/val at top level
            experiments.append({
                "name": d.name,
                "strategy": "decompose",
                "timestamp": meta.get("timestamp", comp.get("timestamp", "?")),
                "model": meta.get("model", {}).get("name", "?"),
                "iters": "-",
                "best_iter": "-",
                "best_train": comp.get("train", {}).get("accuracy", 0),
                "best_val": comp.get("val", {}).get("accuracy", 0),
                "git": meta.get("git_hash", "?"),
            })
        elif strategy == 'decompose-opt':
            # Decompose optimization: has iterations + best tracking
            experiments.append({
                "name": d.name,
                "strategy": "decompose-opt",
                "timestamp": meta.get("timestamp", comp.get("timestamp", "?")),
                "model": meta.get("model", {}).get("name", "?"),
                "iters": comp.get("total_iterations", "?"),
                "best_iter": comp.get("best_iteration", "?"),
                "best_train": comp.get("best", comp.get("train", {})).get("train_accuracy",
                              comp.get("train", {}).get("accuracy", 0)),
                "best_val": comp.get("best", comp.get("val", {})).get("val_accuracy",
                            comp.get("val", {}).get("accuracy", 0)),
                "git": meta.get("git_hash", "?"),
            })
        else:
            # Single-shot format
            experiments.append({
                "name": d.name,
                "strategy": "single-shot",
                "timestamp": meta.get("timestamp", comp.get("timestamp", "?")),
                "model": meta.get("model", {}).get("name", "?"),
                "iters": comp.get("total_iterations", "?"),
                "best_iter": comp.get("best_iteration", "?"),
                "best_train": comp.get("best_claude", {}).get("train_accuracy", 0),
                "best_val": comp.get("best_claude", {}).get("val_accuracy", 0),
                "git": meta.get("git_hash", "?"),
            })

    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'Name':<30} {'Strategy':<12} {'Model':<25} {'Iters':>5} {'Best':>4} "
          f"{'Train':>7} {'Val':>7} {'Git':>8} {'Timestamp':<20}")
    print("-" * 135)
    for e in experiments:
        ts = e["timestamp"][:16] if len(e["timestamp"]) >= 16 else e["timestamp"]
        iters = e['iters']
        best = e['best_iter']
        # Format iters/best as strings to handle both int and "-"
        iters_str = f"{iters:>5}" if isinstance(iters, int) else f"{iters!s:>5}"
        best_str = f"{best:>4}" if isinstance(best, int) else f"{best!s:>4}"
        print(f"{e['name']:<30} {e['strategy']:<12} {e['model']:<25} "
              f"{iters_str} {best_str} {e['best_train']:>7.3f} "
              f"{e['best_val']:>7.3f} {e['git']:>8} {ts:<20}")
    print()


async def run_optimization(config: dict, max_iterations: int,
                           experiment_name: str | None = None):
    """Run the Claude-powered prompt optimization loop."""

    # Determine experiment output directory
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "output" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update latest symlink
    latest_link = Path(__file__).parent / "output" / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(experiment_name)

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output dir: {output_dir.resolve()}")

    # Save metadata for reproducibility
    _save_experiment_metadata(output_dir, config, max_iterations, experiment_name)

    # Load training data
    logger.info("Loading training data...")
    all_examples = load_training_examples(config)

    if len(all_examples) < 4:
        logger.error(f"Only {len(all_examples)} examples — need at least 4")
        return

    # Split train/val (75/25) — same as DSPy pipeline
    split_idx = int(len(all_examples) * 0.75)
    trainset = all_examples[:split_idx]
    valset = all_examples[split_idx:]
    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Setup MedGemma client
    model_config = config['model']
    ollama_client = OpenAI(
        base_url=model_config['api_base'],
        api_key=model_config['api_key'],
    )
    model_name = model_config['name']
    temperature = model_config.get('temperature', 0.1)
    max_tokens = model_config.get('max_tokens', 256)

    opt_config = config.get('optimization', {})
    confidence_threshold = opt_config.get('confidence_threshold', 0.7)
    claude_model = config.get('claude', {}).get('model', 'claude-opus-4-6')

    for iteration in range(max_iterations + 1):
        iter_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}" +
                    (" (baseline)" if iteration == 0 else ""))
        logger.info(f"{'='*60}")

        # Evaluate on training set
        logger.info("Evaluating on training set...")
        train_result = evaluate_prompt(
            current_instruction, current_few_shots, trainset,
            ollama_client, model_name, temperature, max_tokens,
            confidence_threshold,
        )
        logger.info(f"  Train accuracy: {train_result['accuracy']:.3f}")

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_result = evaluate_prompt(
            current_instruction, current_few_shots, valset,
            ollama_client, model_name, temperature, max_tokens,
            confidence_threshold,
        )
        logger.info(f"  Val accuracy: {val_result['accuracy']:.3f}")

        elapsed = time.time() - iter_start

        # Record iteration
        record = {
            'iteration': iteration,
            'instruction': current_instruction,
            'few_shots_count': len(current_few_shots),
            'train_accuracy': train_result['accuracy'],
            'val_accuracy': val_result['accuracy'],
            'accuracy': val_result['accuracy'],  # alias for history summary
            'tp': val_result['tp'],
            'fp': val_result['fp'],
            'tn': val_result['tn'],
            'fn': val_result['fn'],
            'parse_failures': val_result['parse_failures'],
            'elapsed_seconds': round(elapsed, 1),
        }
        iteration_history.append(record)

        # Track best
        if val_result['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_result['accuracy']
            best_iteration = iteration
            # Save best prompt
            (output_dir / "best_prompt.txt").write_text(current_instruction)
            with open(output_dir / "few_shots.json", "w") as f:
                json.dump(current_few_shots, f, indent=2,
                          default=str)
            logger.info(f"  New best! Val accuracy: {best_val_accuracy:.3f}")

        # Print summary
        print(f"\n  Iteration {iteration}: "
              f"train={train_result['accuracy']:.3f} "
              f"val={val_result['accuracy']:.3f} "
              f"({'BEST' if iteration == best_iteration else ''})")

        # Stop if this was the last iteration
        if iteration >= max_iterations:
            break

        # Generate improved prompt via Claude
        logger.info("Generating improved prompt via Claude...")
        error_analysis = build_error_analysis(train_result, trainset)

        try:
            new_instruction, new_few_shots, analysis = await generate_improved_prompt(
                current_instruction, current_few_shots,
                error_analysis, iteration_history, trainset,
                claude_model,
            )
            logger.info(f"Claude analysis: {analysis[:200]}...")
            record['claude_analysis'] = analysis
            current_instruction = new_instruction
            current_few_shots = new_few_shots
        except Exception as e:
            logger.error(f"Claude call failed: {e}")
            logger.info("Keeping current prompt for next iteration")

    # Save full history
    # Strip non-serializable data for JSON output
    serializable_history = []
    for h in iteration_history:
        sh = {k: v for k, v in h.items()
              if k != 'few_shots'}
        serializable_history.append(sh)

    with open(output_dir / "optimization_history.json", "w") as f:
        json.dump(serializable_history, f, indent=2, default=str)

    # Build comparison report
    baseline = iteration_history[0] if iteration_history else {}
    best = iteration_history[best_iteration] if best_iteration is not None else {}

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'total_iterations': max_iterations,
        'best_iteration': best_iteration,
        'baseline': {
            'train_accuracy': baseline.get('train_accuracy', 0),
            'val_accuracy': baseline.get('val_accuracy', 0),
        },
        'best_claude': {
            'train_accuracy': best.get('train_accuracy', 0),
            'val_accuracy': best.get('val_accuracy', 0),
        },
        'dspy_miprov2': {
            'train_accuracy': 0.333,
            'val_accuracy': 0.533,
            'note': 'From previous DSPy MIPROv2 run',
        },
    }

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print final comparison table
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE — COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} {'Train Acc':>10} {'Val Acc':>10}")
    print("-" * 45)
    print(f"{'Baseline (no opt)':<25} "
          f"{baseline.get('train_accuracy', 0):>10.3f} "
          f"{baseline.get('val_accuracy', 0):>10.3f}")
    print(f"{'Claude best (iter ' + str(best_iteration) + ')':<25} "
          f"{best.get('train_accuracy', 0):>10.3f} "
          f"{best.get('val_accuracy', 0):>10.3f}")
    print(f"{'DSPy MIPROv2':<25} {'0.333':>10} {'0.533':>10}")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claude-powered prompt optimization for entity resolution")
    parser.add_argument('--config', default='config.yaml',
                        help='Path to configuration YAML')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of optimization iterations')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: auto-generated timestamp)')
    parser.add_argument('--strategy', choices=['single-shot', 'decompose'],
                        default='single-shot',
                        help='Evaluation strategy (default: single-shot)')
    parser.add_argument('--threshold', type=int, default=None,
                        help='Voting threshold for decompose strategy (default: 3)')
    parser.add_argument('--from', dest='from_experiment', type=str, default=None,
                        help='Warm-start from a previous experiment (e.g. decompose-opt-v1)')
    parser.add_argument('--list', action='store_true',
                        help='List all past experiments and exit')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    if args.list:
        list_experiments()
        return

    config = load_config(args.config)

    # Resolve --from to a best_sub_prompts.json path
    initial_prompts_path = None
    if args.from_experiment:
        candidate = Path(__file__).parent / "output" / args.from_experiment / "best_sub_prompts.json"
        if not candidate.exists():
            parser.error(f"--from: {candidate} does not exist")
        initial_prompts_path = candidate

    if args.strategy == 'decompose':
        if args.iterations and args.iterations > 0:
            # Decompose optimization loop
            anyio.run(run_decompose_optimization, config, args.iterations,
                      args.name, args.threshold, initial_prompts_path)
        else:
            # Single-pass decompose evaluation
            anyio.run(run_decompose_evaluation, config, args.name, args.threshold)
    else:
        if args.iterations is not None:
            config.setdefault('optimization', {})['max_iterations'] = args.iterations
        max_iters = config.get('optimization', {}).get('max_iterations', 8)
        anyio.run(run_optimization, config, max_iters, args.name)


if __name__ == '__main__':
    main()
