from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_env.core import Task
from strands_env.explore import (
    ExplorationPlan,
    ExploredRollout,
    Explorer,
    ExploringSampler,
    HintPoolExplorer,
    LLMExplorer,
)

from .conftest import make_result


class FixedExplorer(Explorer):
    def __init__(self, hints):
        self.hints = hints
        self.calls: list[int] = []

    async def propose(self, task, history, n):
        self.calls.append(len(history))
        if isinstance(self.hints, Exception):
            raise self.hints
        return list(self.hints)


def recording_rollout():
    seen: list[Task] = []

    async def rollout(task):
        seen.append(task)
        return make_result(answer=task.id)

    return rollout, seen


class TestApply:
    def test_string_message(self):
        task = HintPoolExplorer(["h"]).apply(Task(message="q"), "try code")
        assert task.message == "q\n\ntry code"
        assert task.exploration_hint == "try code"

    def test_dict_message_is_not_mutated(self):
        original = Task(message={"role": "user", "content": [{"text": "q"}]})
        task = HintPoolExplorer(["h"]).apply(original, "try code")
        assert task.message["content"] == [{"text": "q"}, {"text": "try code"}]
        assert original.message["content"] == [{"text": "q"}]

    def test_no_hint_is_identity(self):
        task = Task(message="q")
        assert HintPoolExplorer(["h"]).apply(task, None) is task


class TestHintPoolExplorer:
    async def test_least_used_first(self):
        explorer = HintPoolExplorer(["a", "b", "c"])
        history = [ExploredRollout(task=Task(message="q"), hint="a", result=make_result())]
        assert await explorer.propose(Task(message="q"), history, 4) == ["b", "c", "a", "b"]

    def test_rejects_empty_pool(self):
        with pytest.raises(ValueError):
            HintPoolExplorer([])


class TestExploringSampler:
    async def test_unguided_wave_then_guided(self):
        rollout, seen = recording_rollout()
        explorer = FixedExplorer(["h1", "h2"])
        group = await ExploringSampler(rollout, explorer, n_samples=4).sample(Task(id="p", message="q"))

        assert [r.task.id for r in group] == ["p_0", "p_1", "p_2", "p_3"]
        assert [r.hint for r in group] == [None, None, "h1", "h2"]
        assert explorer.calls == [2]
        assert seen[2].message == "q\n\nh1"

    async def test_small_waves_reconsult_explorer(self):
        rollout, _ = recording_rollout()
        explorer = FixedExplorer(["h"])
        sampler = ExploringSampler(rollout, explorer, n_samples=5, n_unguided=1, wave_size=2)
        group = await sampler.sample(Task(id="p", message="q"))

        assert len(group) == 5
        assert explorer.calls == [1, 3]
        # One hint per wave of two: the second slot of each wave falls back to unguided.
        assert [r.hint for r in group] == [None, "h", None, "h", None]

    async def test_fully_guided(self):
        rollout, _ = recording_rollout()
        explorer = FixedExplorer(["a", "b", "c"])
        group = await ExploringSampler(rollout, explorer, n_samples=3, n_unguided=0).sample(Task(message="q"))
        assert [r.hint for r in group] == ["a", "b", "c"]
        assert explorer.calls == [0]

    async def test_explorer_failure_falls_back_to_unguided(self):
        rollout, _ = recording_rollout()
        explorer = FixedExplorer(RuntimeError("boom"))
        group = await ExploringSampler(rollout, explorer, n_samples=2, n_unguided=0).sample(Task(message="q"))
        assert [r.hint for r in group] == [None, None]

    async def test_extra_hints_are_dropped(self):
        rollout, _ = recording_rollout()
        explorer = FixedExplorer(["a", "b", "c"])
        group = await ExploringSampler(rollout, explorer, n_samples=2, n_unguided=0).sample(Task(message="q"))
        assert [r.hint for r in group] == ["a", "b"]


class TestLLMExplorer:
    def test_prompt_hides_ground_truth(self):
        explorer = LLMExplorer(model_factory=MagicMock())
        task = Task(message="What is 6*7?", ground_truth="SECRET")
        history = [ExploredRollout(task=task, hint="guess", result=make_result(reward=0.0))]
        prompt = explorer.build_prompt(task, history, 2)

        assert "SECRET" not in prompt
        assert "What is 6*7?" in prompt
        assert "## Attempt 1 (hint: guess)" in prompt
        assert '"zero_variance"' in prompt
        assert prompt.endswith("Write exactly 2 hints.")

    @patch("strands_env.explore.explorer.Agent")
    async def test_propose_returns_plan_hints(self, mock_agent_cls):
        agent = MagicMock()
        agent.invoke_async = AsyncMock(
            return_value=MagicMock(structured_output=ExplorationPlan(diagnosis="d", hints=["x", "y"]))
        )
        mock_agent_cls.return_value = agent
        explorer = LLMExplorer(model_factory=MagicMock(), instructions="Prefer code.")

        assert await explorer.propose(Task(message="q"), [], 2) == ["x", "y"]
        assert mock_agent_cls.call_args.kwargs["system_prompt"].endswith("Prefer code.")
        assert agent.invoke_async.call_args.kwargs["structured_output_model"] is ExplorationPlan
