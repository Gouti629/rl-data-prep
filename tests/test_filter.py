from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from eval_anything.utils.generation import GenerationResult


def _make_example(idx, question="q", ground_truth="a", **extra_raw):
    raw = {"q": question, "a": ground_truth, **extra_raw}
    return {
        "example_id": f"test_{idx}",
        "question": question,
        "ground_truth": ground_truth,
        "_raw": raw,
    }


ROLLOUTS = 4


def _mock_score(text, gt, example=None):
    if text == "correct":
        return 1.0, {"status": "ok"}
    return 0.0, {"status": "incorrect"}


def _make_gen_fn(correct_map):
    def gen_fn(prompts, **kwargs):
        rollouts = kwargs.get("num_return_sequences", 1)
        results = []
        for prompt in prompts:
            eid = prompt.replace("prompt_", "")
            n = correct_map[eid]
            texts = ["correct"] * n + ["wrong"] * (rollouts - n)
            results.append(GenerationResult(text=texts, token_count=10, truncated=False))
        return results
    return gen_fn


# ---------------------------------------------------------------------------
# RLVR filtering
# ---------------------------------------------------------------------------

class TestFilterForRLVR:
    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_filters_by_pass_rate(self, mock_resolve, mock_eval_cls, mock_load,
                                   mock_make_gen, mock_free, tmp_path):
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(i, f"q{i}", f"a{i}") for i in range(5)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score

        mock_load.return_value = (MagicMock(), MagicMock())

        # ex0: 0/4=0.0, ex1: 1/4=0.25, ex2: 2/4=0.5, ex3: 3/4=0.75, ex4: 4/4=1.0
        correct_map = {"test_0": 0, "test_1": 1, "test_2": 2, "test_3": 3, "test_4": 4}
        mock_make_gen.return_value = _make_gen_fn(correct_map)

        from rl_data_prep.core import filter_for_rlvr

        result_path = filter_for_rlvr(
            model="test-model",
            eval_config="test_eval",
            rollouts=ROLLOUTS,
            min_pass_rate=0.2,
            max_pass_rate=0.8,
            output_dir=str(tmp_path),
        )

        jsonl_path = os.path.join(str(tmp_path), "test-model_test_eval_0.2_0.8", "filtered.jsonl")
        assert os.path.exists(jsonl_path)
        assert result_path == jsonl_path

        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 3
        pass_rates = {r["example_id"]: r["pass_rate"] for r in records}
        assert pass_rates == {"test_1": 0.25, "test_2": 0.5, "test_3": 0.75}

        for rec in records:
            assert rec["rollouts"] == ROLLOUTS
            assert "question" in rec
            assert "ground_truth" in rec
            assert "scores" in rec
            assert "n_correct" in rec

    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_boundary_inclusive(self, mock_resolve, mock_eval_cls, mock_load,
                                mock_make_gen, mock_free, tmp_path):
        """min and max pass_rate bounds are inclusive."""
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(i) for i in range(3)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score

        mock_load.return_value = (MagicMock(), MagicMock())

        # pass_rates: 0.25, 0.5, 0.75
        correct_map = {"test_0": 1, "test_1": 2, "test_2": 3}
        mock_make_gen.return_value = _make_gen_fn(correct_map)

        from rl_data_prep.core import filter_for_rlvr

        filter_for_rlvr(
            model="m", eval_config="x", rollouts=ROLLOUTS,
            min_pass_rate=0.25, max_pass_rate=0.75,
            output_dir=str(tmp_path),
        )

        jsonl_path = os.path.join(str(tmp_path), "m_test_eval_0.25_0.75", "filtered.jsonl")
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 3

    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_none_kept(self, mock_resolve, mock_eval_cls, mock_load,
                       mock_make_gen, mock_free, tmp_path):
        """When no examples fall in range, output is empty."""
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(0)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score

        mock_load.return_value = (MagicMock(), MagicMock())
        # pass_rate = 0.0
        mock_make_gen.return_value = _make_gen_fn({"test_0": 0})

        from rl_data_prep.core import filter_for_rlvr

        filter_for_rlvr(
            model="m", eval_config="x", rollouts=ROLLOUTS,
            min_pass_rate=0.5, max_pass_rate=1.0,
            output_dir=str(tmp_path),
        )

        jsonl_path = os.path.join(str(tmp_path), "m_test_eval_0.5_1.0", "filtered.jsonl")
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 0

    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_summary_json(self, mock_resolve, mock_eval_cls, mock_load,
                          mock_make_gen, mock_free, tmp_path):
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(i) for i in range(3)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_make_gen.return_value = _make_gen_fn({"test_0": 0, "test_1": 2, "test_2": 4})

        from rl_data_prep.core import filter_for_rlvr

        filter_for_rlvr(
            model="m", eval_config="x", rollouts=ROLLOUTS,
            min_pass_rate=0.1, max_pass_rate=0.9,
            output_dir=str(tmp_path),
        )

        summary_path = os.path.join(str(tmp_path), "m_test_eval_0.1_0.9", "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["total_prompts"] == 3
        assert summary["filtered_prompts"] == 1  # only test_1 (0.5)
        assert summary["rollouts"] == ROLLOUTS
        assert summary["min_pass_rate"] == 0.1
        assert summary["max_pass_rate"] == 0.9

    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_raw_fields_included(self, mock_resolve, mock_eval_cls, mock_load,
                                  mock_make_gen, mock_free, tmp_path):
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(0, extra_field="extra_value")]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_make_gen.return_value = _make_gen_fn({"test_0": 2})

        from rl_data_prep.core import filter_for_rlvr

        filter_for_rlvr(
            model="m", eval_config="x", rollouts=ROLLOUTS,
            min_pass_rate=0.0, max_pass_rate=1.0,
            output_dir=str(tmp_path),
        )

        jsonl_path = os.path.join(str(tmp_path), "m_test_eval_0.0_1.0", "filtered.jsonl")
        with open(jsonl_path) as f:
            rec = json.loads(f.readline())

        assert rec["extra_field"] == "extra_value"


# ---------------------------------------------------------------------------
# all_results.jsonl
# ---------------------------------------------------------------------------

class TestAllResults:
    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_all_results_has_every_example(self, mock_resolve, mock_eval_cls, mock_load,
                                           mock_make_gen, mock_free, tmp_path):
        """all_results.jsonl contains every example, not just filtered ones."""
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(i) for i in range(3)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score
        mock_load.return_value = (MagicMock(), MagicMock())

        # pass_rates: 0.0, 0.5, 1.0
        correct_map = {"test_0": 0, "test_1": 2, "test_2": 4}
        mock_make_gen.return_value = _make_gen_fn(correct_map)

        from rl_data_prep.core import filter_for_rlvr

        filter_for_rlvr(
            model="m", eval_config="x", rollouts=ROLLOUTS,
            min_pass_rate=0.5, max_pass_rate=0.5,
            output_dir=str(tmp_path),
        )

        base = os.path.join(str(tmp_path), "m_test_eval_0.5_0.5")

        # filtered.jsonl has only 1 (pass_rate == 0.5)
        with open(os.path.join(base, "filtered.jsonl")) as f:
            filtered = [json.loads(l) for l in f]
        assert len(filtered) == 1

        # all_results.jsonl has all 3 with pass_rate column
        with open(os.path.join(base, "all_results.jsonl")) as f:
            all_recs = [json.loads(l) for l in f]
        assert len(all_recs) == 3

        pass_rates = {r["example_id"]: r["pass_rate"] for r in all_recs}
        assert pass_rates == {"test_0": 0.0, "test_1": 0.5, "test_2": 1.0}

        for rec in all_recs:
            assert "pass_rate" in rec
            assert "question" in rec
            assert "scores" in rec


# ---------------------------------------------------------------------------
# Distillation filtering
# ---------------------------------------------------------------------------

class TestFilterForDistillation:
    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_distill_logic(self, mock_resolve, mock_eval_cls, mock_load,
                           mock_make_gen, mock_free, tmp_path):
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(i, f"q{i}", f"a{i}") for i in range(5)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score

        # Teacher: 0/4=0.0, 2/4=0.5, 3/4=0.75, 4/4=1.0, 4/4=1.0
        teacher_correct = {"test_0": 0, "test_1": 2, "test_2": 3, "test_3": 4, "test_4": 4}
        # Student (on teacher-passed: 1,2,3,4): 1/4=0.25, 3/4=0.75, 0/4=0.0, 2/4=0.5
        student_correct = {"test_1": 1, "test_2": 3, "test_3": 0, "test_4": 2}

        mock_load.side_effect = [
            (MagicMock(), MagicMock()),  # teacher
            (MagicMock(), MagicMock()),  # student
        ]
        mock_make_gen.side_effect = [
            _make_gen_fn(teacher_correct),
            _make_gen_fn(student_correct),
        ]

        from rl_data_prep.core import filter_for_distillation

        result_path = filter_for_distillation(
            student_model="student",
            teacher_model="teacher",
            eval_config="test_eval",
            student_max_pass_rate=0.3,
            teacher_min_pass_rate=0.5,
            rollouts=ROLLOUTS,
            output_dir=str(tmp_path),
        )

        jsonl_path = os.path.join(str(tmp_path), "student_teacher_test_eval_0.5_0.3", "distill_filtered.jsonl")
        assert os.path.exists(jsonl_path)
        assert result_path == jsonl_path

        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]

        # teacher passes: 1 (0.5), 2 (0.75), 3 (1.0), 4 (1.0)
        # student pass_rates: 1->0.25, 2->0.75, 3->0.0, 4->0.5
        # student <= 0.3: examples 1 (0.25) and 3 (0.0)
        assert len(records) == 2
        kept_ids = {r["example_id"] for r in records}
        assert kept_ids == {"test_1", "test_3"}

        for rec in records:
            assert "teacher_pass_rate" in rec
            assert "teacher_scores" in rec
            assert "student_pass_rate" in rec
            assert "student_scores" in rec
            assert rec["rollouts"] == ROLLOUTS

    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_teacher_fails_all(self, mock_resolve, mock_eval_cls, mock_load,
                               mock_make_gen, mock_free, tmp_path):
        """If teacher fails all, student never runs (no prompts), output empty."""
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(0), _make_example(1)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score

        teacher_correct = {"test_0": 0, "test_1": 0}
        mock_load.side_effect = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]
        mock_make_gen.side_effect = [
            _make_gen_fn(teacher_correct),
            _make_gen_fn({}),  # student never called
        ]

        from rl_data_prep.core import filter_for_distillation

        filter_for_distillation(
            student_model="s", teacher_model="t", eval_config="x",
            student_max_pass_rate=1.0, teacher_min_pass_rate=0.5,
            rollouts=ROLLOUTS, output_dir=str(tmp_path),
        )

        jsonl_path = os.path.join(str(tmp_path), "s_t_test_eval_0.5_1.0", "distill_filtered.jsonl")
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 0

    @patch("rl_data_prep.core._free_gpu_memory")
    @patch("rl_data_prep.core._make_generate_fn")
    @patch("rl_data_prep.core._load_model_for_backend")
    @patch("rl_data_prep.core.GenericEval")
    @patch("rl_data_prep.core._resolve_eval_config")
    def test_distill_summary(self, mock_resolve, mock_eval_cls, mock_load,
                             mock_make_gen, mock_free, tmp_path):
        mock_resolve.return_value = {
            "name": "test_eval", "dataset": "test",
            "question_field": "q", "answer_field": "a",
        }

        eval_obj = MagicMock()
        eval_obj.name = "test_eval"
        mock_eval_cls.return_value = eval_obj

        examples = [_make_example(0), _make_example(1)]
        eval_obj.load_data.return_value = examples
        eval_obj.format_prompt.side_effect = lambda ex, tok: f"prompt_{ex['example_id']}"
        eval_obj.score.side_effect = _mock_score

        teacher_correct = {"test_0": 4, "test_1": 4}
        student_correct = {"test_0": 0, "test_1": 4}

        mock_load.side_effect = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]
        mock_make_gen.side_effect = [
            _make_gen_fn(teacher_correct),
            _make_gen_fn(student_correct),
        ]

        from rl_data_prep.core import filter_for_distillation

        filter_for_distillation(
            student_model="s", teacher_model="t", eval_config="x",
            student_max_pass_rate=0.5, teacher_min_pass_rate=0.5,
            rollouts=ROLLOUTS, output_dir=str(tmp_path),
        )

        summary_path = os.path.join(str(tmp_path), "s_t_test_eval_0.5_0.5", "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["total_prompts"] == 2
        assert summary["teacher_passed"] == 2
        assert summary["filtered_prompts"] == 1


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLI:
    @patch("rl_data_prep.cli.filter_for_rlvr")
    def test_rlvr_args(self, mock_filter):
        from rl_data_prep.cli import main

        main([
            "rlvr",
            "--model", "test-model",
            "--eval-config", "gsm8k",
            "--rollouts", "8",
            "--min-pass-rate", "0.1",
            "--max-pass-rate", "0.9",
            "--output-dir", "/tmp/out",
            "--batch-size", "16",
            "--max-new-tokens", "1024",
            "--temperature", "0.5",
            "--backend", "hf",
            "--tensor-parallel-size", "2",
            "--think",
            "--limit", "100",
        ])

        mock_filter.assert_called_once_with(
            model="test-model",
            eval_config="gsm8k",
            rollouts=8,
            min_pass_rate=0.1,
            max_pass_rate=0.9,
            output_dir="/tmp/out",
            batch_size=16,
            max_new_tokens=1024,
            temperature=0.5,
            backend="hf",
            tensor_parallel_size=2,
            think=True,
            limit=100,
        )

    @patch("rl_data_prep.cli.filter_for_rlvr")
    def test_rlvr_defaults(self, mock_filter):
        from rl_data_prep.cli import main

        main([
            "rlvr",
            "--model", "m",
            "--eval-config", "gsm8k",
            "--rollouts", "4",
            "--min-pass-rate", "0.2",
            "--max-pass-rate", "0.8",
            "--output-dir", "/tmp/out",
        ])

        mock_filter.assert_called_once_with(
            model="m",
            eval_config="gsm8k",
            rollouts=4,
            min_pass_rate=0.2,
            max_pass_rate=0.8,
            output_dir="/tmp/out",
            batch_size=8,
            max_new_tokens=512,
            temperature=0.8,
            backend="vllm",
            tensor_parallel_size=1,
            think=False,
            limit=None,
        )

    @patch("rl_data_prep.cli.filter_for_distillation")
    def test_distill_args(self, mock_filter):
        from rl_data_prep.cli import main

        main([
            "distill",
            "--student-model", "student",
            "--teacher-model", "teacher",
            "--eval-config", "math",
            "--student-max-pass-rate", "0.3",
            "--teacher-min-pass-rate", "0.7",
            "--rollouts", "16",
            "--output-dir", "/tmp/out",
        ])

        mock_filter.assert_called_once_with(
            student_model="student",
            teacher_model="teacher",
            eval_config="math",
            student_max_pass_rate=0.3,
            teacher_min_pass_rate=0.7,
            rollouts=16,
            output_dir="/tmp/out",
            batch_size=8,
            max_new_tokens=512,
            temperature=0.8,
            backend="vllm",
            tensor_parallel_size=1,
            think=False,
            limit=None,
        )

    def test_missing_required_args(self):
        from rl_data_prep.cli import main

        with pytest.raises(SystemExit):
            main(["rlvr", "--model", "m"])

    def test_no_subcommand(self):
        from rl_data_prep.cli import main

        with pytest.raises(SystemExit):
            main([])
