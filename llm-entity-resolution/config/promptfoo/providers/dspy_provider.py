"""
Promptfoo custom provider that runs the DSPy MedicalRecordMatcher.

Promptfoo calls this provider via:
    python providers/dspy_provider.py

It reads the prompt (which contains the two medical histories) and
returns the DSPy module's output.

See: https://www.promptfoo.dev/docs/providers/python/
"""

import sys
import json
from pathlib import Path

# Add project paths
_llm_root = str(Path(__file__).resolve().parent.parent.parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
for p in [_llm_root, _project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """
    Promptfoo provider entry point.

    Args:
        prompt: The rendered prompt text (not used directly — we extract vars)
        options: Provider configuration
        context: Contains 'vars' with medical_history_a and medical_history_b

    Returns:
        Dict with 'output' key containing the model response
    """
    import dspy
    from src.dspy_modules import MedicalRecordMatcher
    from src.utils import load_config

    config = load_config(str(Path(_llm_root) / 'config' / 'llm_config.yaml'))

    # Configure DSPy LM
    model_config = config['model']
    model_name = options.get('config', {}).get('model', model_config['name'])

    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=model_config['api_base'],
        api_key=model_config['api_key'],
        temperature=model_config.get('temperature', 0.1),
        max_tokens=model_config.get('max_tokens', 256),
    )
    dspy.configure(lm=lm)

    # Load optimized program if available
    matcher = MedicalRecordMatcher()
    dspy_config = config.get('dspy', {})
    optimized_path = Path(_llm_root) / dspy_config.get(
        'optimized_program', 'data/dspy/optimized_program.json')

    if optimized_path.exists():
        matcher.load(str(optimized_path))

    # Get vars from promptfoo context
    variables = context.get('vars', {})
    history_a = variables.get('medical_history_a', '')
    history_b = variables.get('medical_history_b', '')

    # Run the matcher
    result = matcher(
        medical_history_a=history_a,
        medical_history_b=history_b,
    )

    output = (
        f"reasoning: {result.reasoning}\n"
        f"is_match: {result.is_match}\n"
        f"confidence: {result.confidence}"
    )

    return {'output': output}
