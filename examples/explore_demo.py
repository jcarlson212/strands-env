from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

import click

from strands_env.core.models import ModelConfig, build_model_factory
from strands_env.core.types import Task
from strands_env.environments.math import MathEnv
from strands_env.eval import EvalSample
from strands_env.explore import (
    Explorer,
    ExploringSampler,
    HintPoolExplorer,
    LLMExplorer,
    analyze_group,
    format_report,
    summarize_run,
)

MATH_PROBLEMS = [
    ("How many positive integers n <= 100 make n^2 + n + 41 composite?", "4"),
    ("What is the remainder when 7^2026 is divided by 100?", "49"),
]

HINTS = [
    "Look for a pattern on small cases first, then generalize.",
    "Work modulo small numbers and use periodicity.",
    "Solve it two different ways and check that they agree.",
    "Write down an explicit formula before computing anything.",
]


async def run_demo(
    backend: Literal["sglang", "bedrock"],
    model_id: str | None,
    base_url: str,
    explorer_model_id: str | None,
    n_samples: int,
    output: Path,
) -> None:
    """Sample a group per problem, half unguided and half explorer-guided, and report the spread."""
    policy = build_model_factory(
        ModelConfig(backend=backend, model_id=model_id, base_url=base_url, sampling_params={"max_new_tokens": 8192})
    )
    env = MathEnv(model_factory=policy)

    # A cheap LLM explorer when given one, otherwise a fixed pool of strategies.
    explorer: Explorer
    if explorer_model_id:
        explorer_factory = build_model_factory(ModelConfig(backend="bedrock", model_id=explorer_model_id))
        explorer = LLMExplorer(explorer_factory, instructions="The agent has no tools; steer its reasoning.")
    else:
        explorer = HintPoolExplorer(HINTS)
    sampler = ExploringSampler(env.rollout, explorer, n_samples=n_samples)

    rows: list[str] = []
    groups = {}
    for i, (question, answer) in enumerate(MATH_PROBLEMS):
        group = await sampler.sample(Task(id=f"problem{i}", message=question, ground_truth=answer))
        rows += [EvalSample(task=r.task, result=r.result).model_dump_json() + "\n" for r in group]
        groups[f"problem{i}"] = analyze_group([r.result for r in group])
        for label, subset in (("unguided", [r for r in group if not r.hint]), ("guided", [r for r in group if r.hint])):
            g = analyze_group([r.result for r in subset])
            click.echo(f"problem{i} {label:<9} n={g.n} reward={g.reward_mean} answers={g.distinct_answers}")

    click.echo(format_report(summarize_run(groups), groups))
    # Rows in the `Evaluator` format, so `python -m strands_env.explore` can read the run back.
    await asyncio.to_thread(output.write_text, "".join(rows), encoding="utf-8")
    click.echo(f"Inspect with: python -m strands_env.explore {output} --show problem0")


@click.command()
@click.option("--backend", "-b", required=True, type=click.Choice(["sglang", "bedrock"]), help="Policy backend.")
@click.option("--model-id", default=None, help="Policy model ID. Auto-detected for SGLang if not provided.")
@click.option("--base-url", default="http://localhost:30000", help="Base URL for SGLang server.")
@click.option("--explorer-model-id", default=None, help="Bedrock model for an LLMExplorer; omit to use a hint pool.")
@click.option("--n-samples", default=8, help="Rollouts per problem.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("explore_results.jsonl"))
def main(
    backend: Literal["sglang", "bedrock"],
    model_id: str | None,
    base_url: str,
    explorer_model_id: str | None,
    n_samples: int,
    output: Path,
) -> None:
    """Steer a group of math rollouts with an explorer and compare guided vs unguided samples.

    \b
    python examples/explore_demo.py --backend sglang
    python examples/explore_demo.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514-v1:0 \\
        --explorer-model-id us.anthropic.claude-haiku-4-5-20251001-v1:0
    """
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_demo(backend, model_id, base_url, explorer_model_id, n_samples, output))


if __name__ == "__main__":
    main()
