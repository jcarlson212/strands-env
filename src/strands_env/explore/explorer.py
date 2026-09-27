from __future__ import annotations

import asyncio
import copy
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, ClassVar, Generic, cast, override

from pydantic import BaseModel, Field, SerializeAsAny
from strands import Agent

from strands_env.core.models import ModelFactory
from strands_env.core.types import RolloutResult, Task, TaskT

from .analysis import AnswerFn, analyze_group, normalized_final_response, render_trajectory

logger = logging.getLogger(__name__)


class ExploredRollout(BaseModel, Generic[TaskT]):
    """One rollout of an explored group: the task as sent (hint applied), the hint, and the result."""

    task: SerializeAsAny[TaskT] = Field(..., description="The task the environment ran, hint already applied.")
    hint: str | None = Field(default=None, description="The explorer's guidance; `None` for an unguided rollout.")
    result: RolloutResult


class Explorer(ABC):
    """Proposes per-rollout guidance that steers a group of rollouts away from what it already tried.

    `propose` decides what to say from the group so far; `apply` decides where the hint goes. The
    default puts it at the end of the user message, which every environment that sends
    `task.message` honors.

    Notes:
        A hint in the user message is part of the prompt the policy is trained on. For RL, either
        keep hints strategy-level (never the answer), or strip them before building the training
        sample, which makes the guided rollouts off-policy.

    Example:
        class RetryFailures(Explorer):
            async def propose(self, task, history, n):
                failed = [r for r in history if r.result.reward_result and r.result.reward_result.reward < 1]
                return ["Your first approach may be wrong; verify each step." if failed else None] * n

        sampler = ExploringSampler(env.rollout, RetryFailures(), n_samples=8)
        group = await sampler.sample(task)
    """

    #: Separates the original message from the appended hint.
    hint_separator: str = "\n\n"

    @abstractmethod
    async def propose(self, task: Task, history: Sequence[ExploredRollout[Any]], n: int) -> list[str | None]:
        """Return up to `n` hints for the next rollouts of `task`; `None` leaves a rollout unguided."""

    def apply(self, task: TaskT, hint: str | None) -> TaskT:
        """Return a copy of `task` with `hint` appended to its message and recorded as `exploration_hint`.

        Override for an environment that does not send `task.message` as the first user turn
        (tau2-bench's user simulator replaces it), e.g. to steer through the system prompt instead.
        """
        if hint is None:
            return task
        message = task.message
        if isinstance(message, str):
            message = f"{message}{self.hint_separator}{hint}"
        else:
            message = copy.deepcopy(message)
            message["content"] = [*message["content"], {"text": hint}]
        return task.model_copy(update={"message": message, "exploration_hint": hint})


class HintPoolExplorer(Explorer):
    """Rotates through a fixed pool of strategy hints, least-used first.

    The cheapest explorer: no model call, deterministic, and a baseline to beat before paying for
    an `LLMExplorer`. Pool entries are domain strategies ("work backwards from the goal", "check
    the answer with code"), written once per environment.
    """

    def __init__(self, hints: Sequence[str]) -> None:
        if not hints:
            raise ValueError("HintPoolExplorer needs at least one hint")
        self.hints = list(hints)

    @override
    async def propose(self, task: Task, history: Sequence[ExploredRollout[Any]], n: int) -> list[str | None]:
        uses = Counter(r.hint for r in history)
        proposals: list[str | None] = []
        for _ in range(n):
            hint = min(self.hints, key=lambda h: (uses[h], self.hints.index(h)))
            uses[hint] += 1
            proposals.append(hint)
        return proposals


class ExplorationPlan(BaseModel):
    """Structured output of an `LLMExplorer` call."""

    diagnosis: str = Field(..., description="What the rollouts so far have in common, and where they go wrong.")
    hints: list[str] = Field(
        ...,
        description="One hint per upcoming attempt, each steering toward a distinct strategy not yet tried.",
    )


class LLMExplorer(Explorer):
    """Lightweight LLM agent that reads the group so far and proposes distinct strategies to try next.

    It sees the task, the group's `GroupAnalysis`, and a rendering of the latest trajectories with
    their hints and rewards, but never `task.ground_truth` — a hint that leaks the answer teaches
    the policy nothing. A small, cheap model is usually enough; the hint only has to point.

    Notes:
        A fresh model is built per `propose` call from `model_factory`, matching `Environment`.
    """

    system_prompt: ClassVar[str] = (
        "You are an exploration advisor for an AI agent that attempts the same task several times. "
        "Your job is to make its next attempts cover strategies it has not tried yet, so that at least "
        "one of them succeeds and the attempts differ from each other.\n\n"
        "You will see the task, statistics over the attempts so far, and their trajectories. Look for "
        "mode collapse (every attempt taking the same approach or reaching the same answer), recurring "
        "failure modes (wrong tool use, giving up early, running out of budget), and unused tools.\n\n"
        "Then write one short hint per upcoming attempt. Rules:\n"
        "- Each hint names a concrete approach, in one or two sentences addressed to the agent.\n"
        "- Hints must differ from each other and from hints already used.\n"
        "- Never state or guess the final answer; describe how to work, not what to conclude.\n"
        "- When attempts are failing the same way, steer away from that; when they succeed, still "
        "vary the route so the group stays diverse."
    )

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        instructions: str | None = None,
        max_trajectories: int = 8,
        max_block_chars: int = 300,
        answer_fn: AnswerFn = normalized_final_response,
        reward_threshold: float = 1.0,
    ) -> None:
        """Build an explorer around a (preferably cheap) model.

        Args:
            model_factory: builds the explorer's model, one per `propose` call.
            instructions: domain guidance appended to `system_prompt`, e.g. which tools matter or
                which strategies are known to exist.
            max_trajectories: how many of the most recent rollouts are shown in full; the group
                statistics always cover all of them.
            max_block_chars: per-block clip for rendered trajectories; bounds the explorer's input.
            answer_fn: extracts the answer each rollout committed to, for the answer statistics.
            reward_threshold: reward at which a rollout counts as passing.
        """
        self.model_factory = model_factory
        self.instructions = instructions
        self.max_trajectories = max_trajectories
        self.max_block_chars = max_block_chars
        self.answer_fn = answer_fn
        self.reward_threshold = reward_threshold

    def build_prompt(self, task: Task, history: Sequence[ExploredRollout[Any]], n: int) -> str:
        """Render what the explorer sees. Override to show more or less of the group."""
        message = task.message if isinstance(task.message, str) else json.dumps(task.message["content"], default=str)
        sections = [f"## Task\n{message}"]
        if history:
            analysis = analyze_group(
                [r.result for r in history], answer_fn=self.answer_fn, reward_threshold=self.reward_threshold
            )
            sections.append(f"## Statistics over {len(history)} attempts\n{analysis.model_dump_json(indent=1)}")
            shown = history[-self.max_trajectories :]
            for i, rollout in enumerate(shown, start=len(history) - len(shown) + 1):
                sections.append(
                    f"## Attempt {i} (hint: {rollout.hint or 'none'})\n"
                    f"{render_trajectory(rollout.result, max_block_chars=self.max_block_chars)}"
                )
        else:
            sections.append("## Attempts so far\nNone yet: propose a diverse opening spread of strategies.")
        sections.append(f"Write exactly {n} hints.")
        return "\n\n".join(sections)

    @override
    async def propose(self, task: Task, history: Sequence[ExploredRollout[Any]], n: int) -> list[str | None]:
        system_prompt = f"{self.system_prompt}\n\n{self.instructions}" if self.instructions else self.system_prompt
        agent = Agent(model=self.model_factory(), system_prompt=system_prompt, tools=[], name="Explorer")
        response = await agent.invoke_async(
            self.build_prompt(task, history, n), structured_output_model=ExplorationPlan
        )
        plan = cast(ExplorationPlan, response.structured_output)
        logger.debug("[%s] explorer diagnosis: %s", task.id, plan.diagnosis)
        return list(plan.hints)


class ExploringSampler(Generic[TaskT]):
    """Samples a group of rollouts for one task in waves, letting an `Explorer` steer each wave.

    The first wave runs unguided, so the explorer reacts to how the policy actually behaves; each
    later wave gets hints proposed from every rollout before it. Rollouts within a wave run
    concurrently.

    Notes:
        An explorer that raises, or returns the wrong number of hints, never costs a rollout: the
        missing hints fall back to unguided. An exception from `rollout_fn` propagates.
    """

    def __init__(
        self,
        rollout_fn: Callable[[TaskT], Awaitable[RolloutResult]],
        explorer: Explorer,
        *,
        n_samples: int,
        n_unguided: int | None = None,
        wave_size: int | None = None,
    ) -> None:
        """Configure the waves.

        Args:
            rollout_fn: runs one rollout, e.g. `env.rollout` or `EnvironmentActorPool.rollout`.
            explorer: proposes the hints for every wave after the unguided one.
            n_samples: rollouts per task, the group size.
            n_unguided: size of the unguided first wave; `None` means half the group, `0` guides
                every rollout.
            wave_size: rollouts per guided wave; `None` means one wave for the rest of the group.
                Smaller waves let the explorer react more often, at the cost of less concurrency.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        self.rollout_fn = rollout_fn
        self.explorer = explorer
        self.n_samples = n_samples
        self.n_unguided = n_samples // 2 if n_unguided is None else min(n_unguided, n_samples)
        self.wave_size = max(1, wave_size or n_samples - self.n_unguided)

    async def sample(self, task: TaskT) -> list[ExploredRollout[TaskT]]:
        """Run `n_samples` rollouts of `task`, ids suffixed `_0`, `_1`, ... like `Evaluator`."""
        history: list[ExploredRollout[TaskT]] = []
        while len(history) < self.n_samples:
            remaining = self.n_samples - len(history)
            if not history and self.n_unguided:
                hints: list[str | None] = [None] * self.n_unguided
            else:
                hints = await self.propose_hints(task, history, min(self.wave_size, remaining))
            wave: list[tuple[TaskT, str | None]] = []
            for hint in hints:
                expanded = task.model_copy(deep=True)
                expanded.id = f"{task.id}_{len(history) + len(wave)}"
                wave.append((self.explorer.apply(expanded, hint), hint))
            results = await asyncio.gather(*(self.rollout_fn(t) for t, _ in wave))
            history += [ExploredRollout(task=t, hint=h, result=r) for (t, h), r in zip(wave, results, strict=True)]
        return history

    async def propose_hints(self, task: TaskT, history: Sequence[ExploredRollout[Any]], n: int) -> list[str | None]:
        """Ask the explorer for exactly `n` hints, padding with unguided slots if it falls short or fails."""
        try:
            hints = list(await self.explorer.propose(task, history, n))
        except Exception:
            logger.exception("[%s] explorer failed; running the wave unguided", task.id)
            return [None] * n
        if len(hints) != n:
            logger.warning("[%s] explorer returned %d hints for %d rollouts", task.id, len(hints), n)
        return (hints + [None] * n)[:n]
