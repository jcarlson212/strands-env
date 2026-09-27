import json
from unittest.mock import AsyncMock

from strands_env.core import Task
from strands_env.eval import EvalSample, Evaluator
from strands_env.explore.cli import explore_cmd

from .conftest import make_result


def write_run(tmp_path):
    path = tmp_path / "results.jsonl"
    evaluator = Evaluator(env_factory=AsyncMock(), output_path=path)
    for i, reward in enumerate([0.0, 1.0]):
        task = Task(id=f"p_{i}", message="q", exploration_hint="try code" if i else None)
        evaluator.log_sample(EvalSample(task=task, result=make_result(str(i), reward)))
    return path


def test_report_and_json(runner, tmp_path):
    write_run(tmp_path)
    out = tmp_path / "explore.json"
    result = runner.invoke(explore_cmd, [str(tmp_path), "--json", str(out)])

    assert result.exit_code == 0, result.output
    assert "1 prompts, 2 samples" in result.output
    assert json.loads(out.read_text())["groups"]["p"]["distinct_answers"] == 2


def test_show_prompt(runner, tmp_path):
    path = write_run(tmp_path)
    result = runner.invoke(explore_cmd, [str(path), "--show", "p"])

    assert result.exit_code == 0, result.output
    assert "=== p_1 (hint: try code) ===" in result.output


def test_show_unknown_prompt(runner, tmp_path):
    result = runner.invoke(explore_cmd, [str(write_run(tmp_path)), "--show", "nope"])
    assert result.exit_code != 0
