"""
Langfuse instrumentation for LLM entity resolution.

Sets up tracing so every DSPy/OpenAI call is recorded in Langfuse
for debugging and monitoring.

Usage:
    from src.langfuse_setup import init_langfuse

    init_langfuse(config)  # Call once at startup
    # All subsequent OpenAI calls are traced automatically
"""

import os
import logging

logger = logging.getLogger(__name__)


def init_langfuse(config: dict):
    """
    Initialize Langfuse tracing.

    Sets environment variables and instruments the OpenAI client so
    all calls (including those from DSPy via LiteLLM) are traced.

    Args:
        config: Full configuration dict (reads config['langfuse'])
    """
    lf_config = config.get('langfuse', {})

    if not lf_config.get('enabled', False):
        logger.info("Langfuse disabled in config")
        return

    # Set env vars for Langfuse SDK
    os.environ['LANGFUSE_PUBLIC_KEY'] = lf_config.get('public_key', '')
    os.environ['LANGFUSE_SECRET_KEY'] = lf_config.get('secret_key', '')
    os.environ['LANGFUSE_HOST'] = lf_config.get('base_url', 'http://localhost:3000')

    try:
        from langfuse import Langfuse
        client = Langfuse()
        logger.info(f"Langfuse initialized at {lf_config.get('base_url')}")

        # Instrument OpenAI calls (used by DSPy via LiteLLM)
        _instrument_openai(client)

        return client

    except ImportError:
        logger.warning("langfuse package not installed — tracing disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e} — continuing without tracing")
        return None


def _instrument_openai(client):
    """
    Instrument OpenAI client for automatic tracing.

    Uses openinference if available, falls back to Langfuse's built-in
    OpenAI wrapper.
    """
    try:
        # Try openinference instrumentation (covers DSPy natively)
        from openinference.instrumentation.dspy import DSPyInstrumentor
        DSPyInstrumentor().instrument()
        logger.info("DSPy instrumented via openinference")
    except ImportError:
        # Fallback: Langfuse's OpenAI wrapper
        try:
            from langfuse.openai import openai
            logger.info("OpenAI instrumented via langfuse.openai")
        except ImportError:
            logger.info("No OpenAI instrumentation available")
