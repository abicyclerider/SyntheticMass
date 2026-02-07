"""Utility helpers for LLM entity resolution."""

import re
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_answer(text: str) -> str:
    """
    Strip MedGemma thinking tokens from response.

    MedGemma outputs `<unused94>thought...` then `<unused95>` before the actual answer.
    """
    # Remove thinking block
    cleaned = re.sub(r'<unused94>.*?<unused95>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def get_project_root() -> Path:
    """Return the project root (SyntheticMass/)."""
    return Path(__file__).resolve().parent.parent.parent


def get_run_dir(config: dict) -> Path:
    """Get the full path to the augmentation run directory."""
    return Path(config['base_dir']) / "output" / "augmented" / config['run_id']
