from __future__ import annotations

import gc
import json
import os
from typing import Any, Union

from eval_anything.evals.generic import GenericEval
from eval_anything.runner import _make_generate_fn, load_eval_configs


def _resolve_eval_config(eval_config: Union[dict, str]) -> dict:
    if isinstance(eval_config, dict):
        return eval_config
    if os.path.isfile(eval_config):
        configs = load_eval_configs(eval_config)
        if len(configs) != 1:
            raise ValueError(
                f"Expected 1 eval config in {eval_config}, found {len(configs)}. "
                f"Available: {[c.get('name') for c in configs]}"
            )
        return configs[0]
    all_configs = load_eval_configs("default")
    for cfg in all_configs:
        if cfg["name"] == eval_config:
            return cfg
    available = [c["name"] for c in all_configs]
    raise ValueError(f"Unknown eval config {eval_config!r}. Available: {available}")


def _load_model_for_backend(model_path: str, backend: str, tensor_parallel_size: int):
    if backend == "hf":
        from eval_anything.models.loader_hf import load_model_hf
        return load_model_hf(model_path)
    from eval_anything.models.loader import load_model
    return load_model(model_path, tensor_parallel_size=tensor_parallel_size)


def _free_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _model_short_name(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    return str(obj)


def _compute_pass_rates(eval_obj, data, tokenizer, generate_fn, rollouts, batch_size, max_new_tokens, temperature):
    prompts = [eval_obj.format_prompt(ex, tokenizer) for ex in data]

    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_data = data[i:i + batch_size]

        outputs = generate_fn(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=rollouts,
        )

        for ex, gen_result in zip(batch_data, outputs):
            scores = []
            n_correct = 0
            for text in gen_result.text:
                score_val, _ = eval_obj.score(text, ex["ground_truth"], example=ex)
                scores.append(score_val)
                if score_val >= 1.0:
                    n_correct += 1

            pass_rate = n_correct / rollouts
            results.append({
                "example": ex,
                "pass_rate": pass_rate,
                "n_correct": n_correct,
                "scores": scores,
            })

    return results


def _build_record(example, pass_rate, n_correct, rollouts, scores, extra_fields=None):
    record = {
        "example_id": example["example_id"],
        "question": str(example["question"]),
        "ground_truth": _make_serializable(example["ground_truth"]),
        "pass_rate": pass_rate,
        "n_correct": n_correct,
        "rollouts": rollouts,
        "scores": scores,
    }
    if extra_fields:
        record.update(extra_fields)
    for k, v in example["_raw"].items():
        if k not in record and not k.startswith("_"):
            record[k] = _make_serializable(v)
    return record


def filter_for_rlvr(
    model: str,
    eval_config: Union[dict, str],
    rollouts: int,
    min_pass_rate: float,
    max_pass_rate: float,
    output_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    backend: str = "vllm",
    tensor_parallel_size: int = 1,
    think: bool = False,
    limit: int | None = None,
) -> str:
    config = _resolve_eval_config(eval_config)
    eval_obj = GenericEval(config, think=think)

    llm, tokenizer = _load_model_for_backend(model, backend, tensor_parallel_size)
    generate_fn = _make_generate_fn(backend, llm, tokenizer)

    data = eval_obj.load_data()
    if limit is not None:
        data = data[:limit]
    print(f"Loaded {len(data)} examples for {eval_obj.name}")

    results = _compute_pass_rates(
        eval_obj, data, tokenizer, generate_fn,
        rollouts, batch_size, max_new_tokens, temperature,
    )

    all_records = []
    for r in results:
        record = _build_record(
            r["example"], r["pass_rate"], r["n_correct"], rollouts, r["scores"],
        )
        all_records.append(record)

    filtered = [rec for rec in all_records if min_pass_rate <= rec["pass_rate"] <= max_pass_rate]

    print(f"Filtered {len(filtered)}/{len(data)} examples "
          f"(pass_rate in [{min_pass_rate}, {max_pass_rate}])")

    eval_name = eval_obj.name
    folder_name = f"{_model_short_name(model)}_{eval_name}_{min_pass_rate}_{max_pass_rate}"
    out_path = os.path.join(output_dir, folder_name)
    os.makedirs(out_path, exist_ok=True)

    all_path = os.path.join(out_path, "all_results.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    jsonl_path = os.path.join(out_path, "filtered.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in filtered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "eval_name": eval_name,
        "total_prompts": len(data),
        "filtered_prompts": len(filtered),
        "min_pass_rate": min_pass_rate,
        "max_pass_rate": max_pass_rate,
        "rollouts": rollouts,
        "model": model,
        "backend": backend,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "think": think,
        "limit": limit,
    }
    summary_path = os.path.join(out_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    del llm, tokenizer
    _free_gpu_memory()

    print(f"Saved to {jsonl_path}")
    return jsonl_path


def filter_for_distillation(
    student_model: str,
    teacher_model: str,
    eval_config: Union[dict, str],
    student_max_pass_rate: float,
    teacher_min_pass_rate: float,
    rollouts: int,
    output_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    backend: str = "vllm",
    tensor_parallel_size: int = 1,
    think: bool = False,
    limit: int | None = None,
) -> str:
    config = _resolve_eval_config(eval_config)
    eval_obj = GenericEval(config, think=think)

    data = eval_obj.load_data()
    if limit is not None:
        data = data[:limit]
    print(f"Loaded {len(data)} examples for {eval_obj.name}")

    # --- Teacher ---
    print(f"Running teacher: {teacher_model}")
    teacher_llm, teacher_tok = _load_model_for_backend(teacher_model, backend, tensor_parallel_size)
    teacher_gen = _make_generate_fn(backend, teacher_llm, teacher_tok)
    teacher_results = _compute_pass_rates(
        eval_obj, data, teacher_tok, teacher_gen,
        rollouts, batch_size, max_new_tokens, temperature,
    )

    teacher_passed_data = []
    teacher_result_map: dict[str, dict] = {}
    for r in teacher_results:
        if r["pass_rate"] >= teacher_min_pass_rate:
            teacher_passed_data.append(r["example"])
            teacher_result_map[r["example"]["example_id"]] = r

    print(f"Teacher passed: {len(teacher_passed_data)}/{len(data)} "
          f"(pass_rate >= {teacher_min_pass_rate})")

    del teacher_llm, teacher_tok
    _free_gpu_memory()

    # --- Student ---
    print(f"Running student: {student_model}")
    student_llm, student_tok = _load_model_for_backend(student_model, backend, tensor_parallel_size)
    student_gen = _make_generate_fn(backend, student_llm, student_tok)
    student_results = _compute_pass_rates(
        eval_obj, teacher_passed_data, student_tok, student_gen,
        rollouts, batch_size, max_new_tokens, temperature,
    )

    all_student_records = []
    filtered = []
    for sr in student_results:
        eid = sr["example"]["example_id"]
        tr = teacher_result_map[eid]
        record = _build_record(
            sr["example"], sr["pass_rate"], sr["n_correct"], rollouts, sr["scores"],
            extra_fields={
                "teacher_pass_rate": tr["pass_rate"],
                "teacher_n_correct": tr["n_correct"],
                "teacher_scores": tr["scores"],
                "student_pass_rate": sr["pass_rate"],
                "student_n_correct": sr["n_correct"],
                "student_scores": sr["scores"],
            },
        )
        all_student_records.append(record)
        if sr["pass_rate"] <= student_max_pass_rate:
            filtered.append(record)

    print(f"Distillation filtered: {len(filtered)}/{len(teacher_passed_data)} "
          f"(student pass_rate <= {student_max_pass_rate})")

    eval_name = eval_obj.name
    folder_name = (
        f"{_model_short_name(student_model)}_{_model_short_name(teacher_model)}"
        f"_{eval_name}_{teacher_min_pass_rate}_{student_max_pass_rate}"
    )
    out_path = os.path.join(output_dir, folder_name)
    os.makedirs(out_path, exist_ok=True)

    all_path = os.path.join(out_path, "all_results.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for rec in all_student_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    jsonl_path = os.path.join(out_path, "distill_filtered.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in filtered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "eval_name": eval_name,
        "total_prompts": len(data),
        "teacher_passed": len(teacher_passed_data),
        "filtered_prompts": len(filtered),
        "teacher_min_pass_rate": teacher_min_pass_rate,
        "student_max_pass_rate": student_max_pass_rate,
        "rollouts": rollouts,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "backend": backend,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "think": think,
        "limit": limit,
    }
    summary_path = os.path.join(out_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    del student_llm, student_tok
    _free_gpu_memory()

    print(f"Saved to {jsonl_path}")
    return jsonl_path
