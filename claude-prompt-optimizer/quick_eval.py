"""Quick evaluation script to eyeball raw MedGemma responses."""
import sys, os, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm-entity-resolution'))
from optimize import load_training_examples, assemble_prompt
from openai import OpenAI

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))
config['dataset']['n_match'] = 3
config['dataset']['n_nonmatch'] = 3

examples = load_training_examples(config)
matches = sum(1 for e in examples if e['is_match'])
print(f'Loaded {len(examples)} examples: {matches} match, {len(examples)-matches} non-match')
sys.stdout.flush()

instruction = """You are comparing medical records from two patients at different hospitals to determine if they are the SAME person.

Think step by step before answering:
1. List conditions that appear in BOTH patients
2. List medications that appear in BOTH patients
3. Check if encounter timelines are compatible (same person can visit two hospitals)
4. Look for any contradictions (e.g. incompatible conditions)

IMPORTANT: Many patients ARE the same person seen at different facilities. Do not default to "false" — carefully weigh the evidence for AND against a match.

After your analysis, respond with:
- is_match: true or false
- confidence: a decimal between 0.0 and 1.0"""

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
model = 'medgemma:1.5-4b'

correct = 0
for i, ex in enumerate(examples):
    prompt = assemble_prompt(instruction, [], ex['history_a'], ex['history_b'])
    if i == 0:
        print(f'Prompt length: {len(prompt)} chars')

    resp = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content or ''
    actual = 'MATCH' if ex['is_match'] else 'NON-MATCH'

    # Quick parse (without stripping thinking)
    raw_lower = raw.lower()
    if 'is_match: true' in raw_lower or 'is_match:true' in raw_lower:
        predicted = 'MATCH'
    elif 'is_match: false' in raw_lower or 'is_match:false' in raw_lower:
        predicted = 'NON-MATCH'
    else:
        predicted = 'UNPARSED'
    tag = 'OK' if predicted == actual else 'WRONG'
    if predicted == actual:
        correct += 1

    print(f'\n===== Example {i} (actual={actual}, predicted={predicted}) [{tag}] =====')
    print(f'Record IDs: {ex["record_id_1"]} vs {ex["record_id_2"]}')
    print(f'--- RAW RESPONSE ---')
    print(raw)
    print(f'--- END ---')
    sys.stdout.flush()

print(f'\n===== SUMMARY =====')
print(f'Correct: {correct}/{len(examples)} = {correct/len(examples):.1%}')
