# rl-data-prep

Filter evaluation datasets by model pass rate for RL and distillation post-training.

Uses [eval-anything](https://github.com/Gouti629/eval-anything) for dataset loading, prompt formatting, generation, and scoring. Works with all eval-anything built-in configs: gsm8k, math, mbpp, humaneval, livecodebench, mmlu, arc.

## Install

```bash
pip install git+https://github.com/Gouti629/rl-data-prep.git
```

For vLLM backend support:

```bash
pip install "rl-data-prep[vllm] @ git+https://github.com/Gouti629/rl-data-prep.git"
```

## Usage

### RLVR filtering

Keep prompts where the model's pass rate falls within a target window — not too easy, not too hard.

```bash
rl-data-prep rlvr \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --eval-config gsm8k \
  --rollouts 8 \
  --min-pass-rate 0.1 \
  --max-pass-rate 0.9 \
  --output-dir ./filtered
```

Output:

```
./filtered/Llama-3.1-8B-Instruct_gsm8k_0.1_0.9/
├── all_results.jsonl   # every example with its pass_rate
├── filtered.jsonl      # only examples with pass_rate in [0.1, 0.9]
└── summary.json
```

`all_results.jsonl` has every example with a `pass_rate` column, so you can re-filter offline without re-running generation:

```python
import pandas as pd
df = pd.read_json("all_results.jsonl", lines=True)
hard = df[df.pass_rate.between(0.0, 0.3)]
```

Each JSONL record contains:

```json
{
  "example_id": "gsm8k_0_abc123",
  "question": "...",
  "ground_truth": "...",
  "pass_rate": 0.375,
  "n_correct": 3,
  "rollouts": 8,
  "scores": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}
```

### Distillation filtering

Keep prompts the teacher solves but the student can't — ideal for distillation training data.

```bash
rl-data-prep distill \
  --teacher-model meta-llama/Llama-3.1-70B-Instruct \
  --student-model meta-llama/Llama-3.1-8B-Instruct \
  --eval-config math \
  --teacher-min-pass-rate 0.75 \
  --student-max-pass-rate 0.25 \
  --rollouts 8 \
  --output-dir ./filtered
```

Output follows the same structure — `all_results.jsonl` (all teacher-passed examples with student pass rates), `distill_filtered.jsonl`, and `summary.json`. The teacher runs first, then is unloaded to free GPU memory before the student runs on teacher-passed prompts only.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | 8 | Prompts per generation batch |
| `--max-new-tokens` | 512 | Max tokens per completion |
| `--temperature` | 0.8 | Sampling temperature |
| `--backend` | vllm | `vllm` or `hf` |
| `--tensor-parallel-size` | 1 | GPUs for tensor parallelism (vLLM) |
| `--think` | false | Allow chain-of-thought `<think>` tags |
| `--limit` | none | Cap number of dataset examples |

### Python API

```python
from rl_data_prep import filter_for_rlvr, filter_for_distillation

# RLVR
filter_for_rlvr(
    model="meta-llama/Llama-3.1-8B-Instruct",
    eval_config="gsm8k",
    rollouts=8,
    min_pass_rate=0.1,
    max_pass_rate=0.9,
    output_dir="./filtered",
)

# Distillation
filter_for_distillation(
    student_model="meta-llama/Llama-3.1-8B-Instruct",
    teacher_model="meta-llama/Llama-3.1-70B-Instruct",
    eval_config="math",
    student_max_pass_rate=0.25,
    teacher_min_pass_rate=0.75,
    rollouts=8,
    output_dir="./filtered",
)
```

The `eval_config` parameter accepts a built-in name (`gsm8k`, `math`, `mbpp`, `humaneval`, `livecodebench_v5`, `mmlu`, `arc`), a path to a YAML config file, or a config dict.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests mock generation and scoring — no GPU required.
