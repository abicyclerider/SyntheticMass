# MedGemma Local Deployment via Ollama

Local LLM serving setup for MedGemma models, used as an independent API for the
SyntheticMass project. The Ollama server runs separately from the project code
and exposes an OpenAI-compatible REST API at `http://localhost:11434/v1/`.

## Hardware

- MacBook Pro 14-inch (Nov 2023)
- Apple M3 Pro
- 18 GB unified memory
- macOS Tahoe 26.2
- Ollama 0.14.2

## Available Models

Four MedGemma 1.5 4B variants are available:

| Model Tag | Size | Thinking | Gen Speed | Notes |
|-----------|------|----------|-----------|-------|
| `medgemma:1.5-4b-q4` | 2.5 GB | Yes | ~39 tok/s | Default Q4, includes CoT reasoning |
| `medgemma:1.5-4b-q4-fast` | 2.5 GB | **No** | ~39 tok/s | Q4 thinking suppressed via `<unused95>` prefill |
| `medgemma:1.5-4b` | 8.6 GB | Yes | ~15 tok/s | Higher precision, tighter on 18GB |
| `medgemma:1.5-4b-fast` | 8.6 GB | **No** | ~15 tok/s | Full precision thinking suppressed via no system prompt |

The **`-fast` variants** suppress chain-of-thought reasoning so the model answers
directly, generating ~10x fewer tokens per response.

The 27B text-only model (~17 GB) does not fit on 18 GB RAM alongside macOS.

### Model Selection Guidance

- **Use `medgemma:1.5-4b-q4-fast`** for iterative development and batch
  processing where speed matters most. Fastest option overall.
- **Use `medgemma:1.5-4b-fast`** for higher quality answers without thinking
  overhead. Best quality-to-speed ratio.
- **Use the non-fast variants** when you want the full chain-of-thought
  reasoning (debugging, understanding model logic, auditing).

### Building the Fast Variants

```bash
# Q4 fast (uses <unused95> prefill in template)
ollama create medgemma:1.5-4b-q4-fast -f ollama/Modelfile.no-thinking

# Full precision fast (uses blob reference to avoid inherited system prompt)
ollama create medgemma:1.5-4b-fast -f ollama/Modelfile.no-thinking-full
```

## Quick Start

```bash
# Start Ollama (if not already running as a background service)
ollama serve

# Verify models are available
ollama list

# Quick test
ollama run medgemma:1.5-4b-q4 "What are common causes of elevated troponin?"
```

## OpenAI-Compatible API

Ollama natively exposes an OpenAI-compatible API. No additional setup required.

### Endpoint

```
Base URL: http://localhost:11434/v1/
```

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by SDK but not validated
)

response = client.chat.completions.create(
    model="medgemma:1.5-4b-q4-fast",
    messages=[{"role": "user", "content": "Your prompt here"}],
    temperature=0.3,
)

print(response.choices[0].message.content)
```

### curl

```bash
curl -s -X POST http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "medgemma:1.5-4b-q4-fast",
    "messages": [{"role": "user", "content": "Your prompt here"}],
    "temperature": 0.3
  }'
```

### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions |
| `POST /v1/completions` | Text completions |

## MedGemma Thinking Tokens

The base MedGemma models output chain-of-thought reasoning wrapped in special
tokens before the actual answer:

```
<unused94>thought
... internal reasoning (often 500+ tokens) ...
<unused95>
... actual response ...
```

**To avoid this entirely**, use the `-fast` model variants (see above).

### What Triggers Thinking

Investigation revealed two independent mechanisms:

1. **System prompt** — On the full precision model, any system prompt in the
   Modelfile (even "answer directly") triggers thinking. Removing the system
   prompt suppresses it completely. This is how `Modelfile.no-thinking-full`
   works.

2. **Template prefill** — On the Q4 model, pre-filling `<unused95>` in the
   response template works reliably to suppress thinking even with a system
   prompt present. This is how `Modelfile.no-thinking` works. (This approach
   does NOT work on the full precision model — the Go template engine
   tokenizes `<unused95>` as literal characters rather than the special token.)

### Parsing Thinking Tokens

If you use the base models and need to parse out the thinking:

```python
def extract_answer(content: str) -> str:
    """Extract the final answer from MedGemma output, stripping thinking tokens."""
    marker = "<unused95>"
    if marker in content:
        return content.split(marker, 1)[1].strip()
    return content.strip()
```

## Test Results (2026-02-06)

### Test 1: Medical Knowledge (both base models)

**Prompt:** "What are the common causes of elevated troponin levels? Answer
concisely in 3-4 sentences."

**medgemma:1.5-4b (8.6 GB)**
- Total time: 60.3s (8.7s load + 50.4s generation)
- Tokens generated: 760
- Speed: 15.07 tok/s
- Response quality: Accurate, well-structured. Listed MI, ACS, myocarditis,
  PE, rhabdomyolysis, sepsis, aortic dissection, medications.

**medgemma:1.5-4b-q4 (2.5 GB)**
- Total time: 20.2s (3.8s load + 15.5s generation)
- Tokens generated: 605
- Speed: 39.16 tok/s
- Response quality: Accurate, slightly different set of causes. Listed MI,
  myocarditis, arrhythmias, PE, pericarditis, kidney disease.

Both models produced medically accurate and relevant responses.

### Test 2: Entity Resolution via API (Q4 model)

**Prompt:** Entity resolution task comparing two patient records with minor
variations (Jonathan/Jon, Main St/Main Street).

**Result:** Correctly identified the records as the same person. Reasoning
cited identical DOB and SSN as strongest indicators, with similar names and
addresses as supporting evidence. Response was well-structured and logical.

### Test 3: No-Thinking Speed Comparison (Q4)

| Metric | Q4 (with thinking) | Q4-fast (no thinking) |
|--------|--------------------|-----------------------|
| Wall time | 13.5s | **1.5s** |
| Completion tokens | 563 | **54** |
| Speedup | baseline | **~9x faster** |

### Test 4: No-Thinking Speed Comparison (Full Precision)

| Metric | 1.5-4b (with thinking) | 1.5-4b-fast (no thinking) |
|--------|------------------------|---------------------------|
| Wall time | 46.1s | **15.0s** |
| Completion tokens | 645 | **63** |
| Speedup | baseline | **~3x wall time, ~10x fewer tokens** |

Note: Wall time includes model swap overhead on 18GB RAM. True generation
speedup is ~10x (proportional to token reduction).

### Benchmark Summary

| Metric | 1.5-4b | 1.5-4b-fast | 1.5-4b-q4 | 1.5-4b-q4-fast |
|--------|--------|-------------|-----------|----------------|
| Generation | 15.1 tok/s | 15.1 tok/s | 39.2 tok/s | 39.2 tok/s |
| Tokens per response | ~650 | **~65** | ~560 | **~55** |
| RAM usage | ~8-10 GB | ~8-10 GB | ~3-4 GB | ~3-4 GB |
| Thinking | Yes | **No** | Yes | **No** |

## Model Sources

These are community-packaged models on Ollama (not official Google releases):

- `medgemma:1.5-4b` — from Ollama community
- `medgemma:1.5-4b-q4` — Q4 quantized variant

The `-fast` variants are built locally from custom Modelfiles in this directory.

Official model weights are on Hugging Face at `google/medgemma-1.5-4b-it`
(requires accepting Google's Health AI Developer Foundations terms of use).

## Troubleshooting

```bash
# Check if Ollama is running
curl -s http://localhost:11434/v1/models | python3 -m json.tool

# Restart Ollama
killall ollama && ollama serve

# Check memory usage
memory_pressure
```
