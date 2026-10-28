from __future__ import annotations

import argparse

from rl_data_prep.core import filter_for_distillation, filter_for_rlvr


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="rl-data-prep")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- rlvr --
    rlvr = subparsers.add_parser("rlvr", help="Filter prompts for RLVR training")
    rlvr.add_argument("--model", required=True)
    rlvr.add_argument("--eval-config", required=True, help="YAML path or built-in name (gsm8k, math, ...)")
    rlvr.add_argument("--rollouts", type=int, required=True)
    rlvr.add_argument("--min-pass-rate", type=float, required=True)
    rlvr.add_argument("--max-pass-rate", type=float, required=True)
    rlvr.add_argument("--output-dir", required=True)
    rlvr.add_argument("--batch-size", type=int, default=8)
    rlvr.add_argument("--max-new-tokens", type=int, default=512)
    rlvr.add_argument("--temperature", type=float, default=0.8)
    rlvr.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    rlvr.add_argument("--tensor-parallel-size", type=int, default=1)
    rlvr.add_argument("--think", action="store_true")
    rlvr.add_argument("--limit", type=int, default=None)

    # -- distill --
    distill = subparsers.add_parser("distill", help="Filter prompts for distillation")
    distill.add_argument("--student-model", required=True)
    distill.add_argument("--teacher-model", required=True)
    distill.add_argument("--eval-config", required=True, help="YAML path or built-in name (gsm8k, math, ...)")
    distill.add_argument("--student-max-pass-rate", type=float, required=True)
    distill.add_argument("--teacher-min-pass-rate", type=float, required=True)
    distill.add_argument("--rollouts", type=int, required=True)
    distill.add_argument("--output-dir", required=True)
    distill.add_argument("--batch-size", type=int, default=8)
    distill.add_argument("--max-new-tokens", type=int, default=512)
    distill.add_argument("--temperature", type=float, default=0.8)
    distill.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    distill.add_argument("--tensor-parallel-size", type=int, default=1)
    distill.add_argument("--think", action="store_true")
    distill.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == "rlvr":
        filter_for_rlvr(
            model=args.model,
            eval_config=args.eval_config,
            rollouts=args.rollouts,
            min_pass_rate=args.min_pass_rate,
            max_pass_rate=args.max_pass_rate,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
            think=args.think,
            limit=args.limit,
        )
    elif args.command == "distill":
        filter_for_distillation(
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            eval_config=args.eval_config,
            student_max_pass_rate=args.student_max_pass_rate,
            teacher_min_pass_rate=args.teacher_min_pass_rate,
            rollouts=args.rollouts,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
            think=args.think,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
