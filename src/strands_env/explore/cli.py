from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from strands_env.utils.loader import load_function

from .analysis import (
    analyze_group,
    format_report,
    load_eval_results,
    normalized_final_response,
    render_trajectory,
    summarize_run,
)

logger = logging.getLogger(__name__)


@click.command("explore")
@click.argument("results_path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option(
    "--answer-fn",
    type=str,
    default=None,
    help="Dotted path to a callable (RolloutResult) -> str | None that extracts the answer. Default: final response.",
)
@click.option("--reward-threshold", type=float, default=1.0, help="Reward at which a sample counts as passing.")
@click.option("--top", type=int, default=10, help="How many of the least diverse prompts to list.")
@click.option("--show", "show_prompt", type=str, default=None, help="Print every trajectory of this prompt id.")
@click.option("--max-block-chars", type=int, default=400, help="Per-block clip when printing trajectories.")
@click.option("--include-aborted", is_flag=True, default=False, help="Include samples marked aborted.")
@click.option(
    "--json", "json_path", type=click.Path(path_type=Path), default=None, help="Also write the analysis as JSON here."
)
def explore_cmd(
    results_path: Path,
    answer_fn: str | None,
    reward_threshold: float,
    top: int,
    show_prompt: str | None,
    max_block_chars: int,
    include_aborted: bool,
    json_path: Path | None,
) -> None:
    """Inspect how diverse the rollouts in an eval run are.

    RESULTS_PATH is a `results.jsonl` written by `Evaluator`, or the run directory holding it.

    Examples:
        python -m strands_env.explore runs/aime/results.jsonl
        python -m strands_env.explore runs/aime --show aime_2024_7 --json runs/aime/explore.json
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if results_path.is_dir():
        results_path = results_path / "results.jsonl"
    samples = load_eval_results(results_path, include_aborted=include_aborted)
    extract = load_function(answer_fn) if answer_fn else normalized_final_response

    groups = {
        prompt_id: analyze_group(
            [s.result for s in prompt_samples], answer_fn=extract, reward_threshold=reward_threshold
        )
        for prompt_id, prompt_samples in samples.items()
    }
    summary = summarize_run(groups, reward_threshold=reward_threshold)
    click.echo(format_report(summary, groups, top=top))

    if show_prompt is not None:
        if show_prompt not in samples:
            raise click.BadParameter(f"no prompt id {show_prompt!r} in {results_path}", param_hint="--show")
        for sample in samples[show_prompt]:
            hint = getattr(sample.task, "exploration_hint", None)
            click.echo(f"\n=== {sample.task.id}" + (f" (hint: {hint})" if hint else "") + " ===")
            click.echo(render_trajectory(sample.result, max_block_chars=max_block_chars))

    if json_path is not None:
        payload = {"summary": summary.model_dump(), "groups": {pid: g.model_dump() for pid, g in groups.items()}}
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote analysis to %s", json_path)
