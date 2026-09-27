import json
from unittest.mock import AsyncMock

import pytest

from strands_env.core import RolloutResult, Task
from strands_env.eval import EvalSample, Evaluator
from strands_env.explore import (
    analyze_group,
    format_report,
    lexical_diversity,
    load_eval_results,
    render_trajectory,
    summarize_run,
    tool_sequence,
)

from .conftest import make_result


def test_tool_sequence_keeps_call_order():
    assert tool_sequence(make_result(tools=("search", "scrape", "search"))) == ("search", "scrape", "search")


class TestLexicalDiversity:
    def test_identical_texts(self):
        assert lexical_diversity(["a b c", "A b C"]) == 0.0

    def test_disjoint_texts(self):
        assert lexical_diversity(["a b", "c d"]) == 1.0

    def test_single_text(self):
        assert lexical_diversity(["a b"]) is None


class TestRenderTrajectory:
    def test_shows_tools_and_outcome(self):
        text = render_trajectory(make_result(tools=("search",), reward=0.0))
        assert "[tool_call] search(" in text
        assert "[tool_result:success] ok" in text
        assert "[assistant] The answer is 42" in text
        assert text.endswith("[outcome] reward=0 termination=task_complete")

    def test_clips_long_blocks_from_the_middle(self):
        text = render_trajectory(make_result(answer="x" * 1000), max_block_chars=40)
        assert "chars...]" in text
        assert text.splitlines()[0].endswith("x" * 20)


class TestAnalyzeGroup:
    def test_collapsed_group(self):
        g = analyze_group([make_result() for _ in range(4)])
        assert g.zero_variance
        assert g.distinct_answers == 1
        assert g.mode_answer_share == 1.0
        assert g.lexical_diversity == 0.0
        assert g.pass_rate == 1.0

    def test_diverse_group(self):
        g = analyze_group(
            [
                make_result("1", 0.0, tools=("a",)),
                make_result("2", 1.0, tools=("b", "a")),
                make_result("3", 0.5),
            ]
        )
        assert not g.zero_variance
        assert g.distinct_answer_ratio == 1.0
        assert g.distinct_tool_sequences == 3
        assert g.tool_usage == {"a": 2, "b": 1}
        assert g.reward_mean == 0.5
        assert g.pass_rate == pytest.approx(1 / 3, abs=1e-4)

    def test_unscored_rollouts_leave_reward_stats(self):
        g = analyze_group([make_result(reward=None), make_result(reward=1.0)])
        assert g.n_unscored == 1
        assert g.reward_mean == 1.0
        assert not g.zero_variance

    def test_custom_answer_fn(self):
        g = analyze_group([make_result("1"), make_result("2")], answer_fn=lambda r: "same")
        assert g.distinct_answers == 1

    def test_empty_group(self):
        assert analyze_group([]).n == 0

    def test_counts_terminations(self):
        g = analyze_group([RolloutResult(), make_result()])
        assert g.termination_reasons == {"not_terminated": 1, "task_complete": 1}


def test_summarize_run_splits_zero_variance():
    groups = {
        "easy": analyze_group([make_result(reward=1.0)] * 2),
        "hard": analyze_group([make_result(reward=0.0)] * 2),
        "mixed": analyze_group([make_result(reward=0.0), make_result(reward=1.0)]),
        "empty": analyze_group([]),
    }
    summary = summarize_run(groups)
    assert summary.n_prompts == 3
    assert summary.zero_variance_share == pytest.approx(2 / 3, abs=1e-4)
    assert summary.all_pass_share == pytest.approx(1 / 3, abs=1e-4)
    assert summary.all_fail_share == pytest.approx(1 / 3, abs=1e-4)
    assert "3 prompts, 6 samples" in format_report(summary, groups)


def test_load_eval_results_reads_evaluator_output(tmp_path):
    path = tmp_path / "results.jsonl"
    evaluator = Evaluator(env_factory=AsyncMock(), output_path=path)
    evaluator.log_sample(EvalSample(task=Task(id="p_0", message="q"), result=make_result()))
    evaluator.log_sample(EvalSample(task=Task(id="p_1", message="q"), result=make_result(), aborted=True))
    before = path.read_text()

    assert [s.task.id for s in load_eval_results(path)["p"]] == ["p_0"]
    assert len(load_eval_results(path, include_aborted=True)["p"]) == 2
    assert path.read_text() == before
    assert json.loads(before.splitlines()[0])["task"]["id"] == "p_0"
