from __future__ import annotations

import base64
import itertools
import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from strands_env.core.types import RolloutResult, extract_message_text
from strands_env.eval.evaluator import EvalSample

#: Maps a rollout to the answer it committed to; `None` means it gave none.
type AnswerFn = Callable[[RolloutResult], str | None]

_WORD_PATTERN = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Per-rollout views
# ---------------------------------------------------------------------------


def tool_sequence(result: RolloutResult) -> tuple[str, ...]:
    """Return the names of the tools the agent called, in call order."""
    return tuple(
        block["toolUse"]["name"]
        for message in result.messages
        if message.get("role") == "assistant"
        for block in message.get("content") or []
        if isinstance(block, dict) and "toolUse" in block
    )


def normalized_final_response(result: RolloutResult) -> str | None:
    """Return the final response lowercased with whitespace collapsed, the default `AnswerFn`."""
    response = result.final_response
    return " ".join(response.lower().split()) or None if response else None


def render_trajectory(result: RolloutResult, *, max_block_chars: int = 400) -> str:
    """Render a rollout as compact plain text, one line per content block.

    Long blocks are cut to `max_block_chars` from the middle, keeping both ends: the start says
    what the agent set out to do and the end what it concluded.
    """

    def clip(text: str) -> str:
        text = " ".join(text.split())
        if len(text) <= max_block_chars:
            return text
        half = max_block_chars // 2
        return f"{text[:half]} [...{len(text) - 2 * half} chars...] {text[-half:]}"

    lines: list[str] = []
    for message in result.messages:
        role = message.get("role", "?")
        for block in message.get("content") or []:
            if "text" in block:
                lines.append(f"[{role}] {clip(block['text'])}")
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                lines.append(f"[tool_call] {tool_use['name']}({clip(json.dumps(tool_use.get('input'), default=str))})")
            elif "toolResult" in block:
                tool_result = block["toolResult"]
                text = " ".join(c["text"] for c in tool_result.get("content") or [] if "text" in c)
                lines.append(f"[tool_result:{tool_result.get('status', '?')}] {clip(text)}")
    reward = f"{result.reward_result.reward:g}" if result.reward_result else "n/a"
    lines.append(f"[outcome] reward={reward} termination={result.termination_reason.value}")
    return "\n".join(lines)


def lexical_diversity(texts: Sequence[str]) -> float | None:
    """Return one minus the mean pairwise Jaccard similarity of the texts' word sets.

    0.0 means every text uses the same vocabulary, 1.0 means no two share a word. `None` below two
    texts, where there is no pair to compare.
    """
    word_sets = [set(_WORD_PATTERN.findall(text.lower())) for text in texts]
    if len(word_sets) < 2:
        return None
    similarities = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in itertools.combinations(word_sets, 2)]
    return round(1.0 - statistics.fmean(similarities), 4)


# ---------------------------------------------------------------------------
# Group analysis (the k rollouts of one prompt)
# ---------------------------------------------------------------------------


class GroupAnalysis(BaseModel):
    """How the rollouts sampled for one prompt spread across outcomes, answers, and strategies."""

    n: int = Field(..., description="Rollouts in the group.")
    n_unscored: int = Field(default=0, description="Rollouts without a `reward_result`; excluded from reward stats.")
    reward_mean: float | None = None
    reward_std: float | None = Field(default=None, description="Population standard deviation over scored rollouts.")
    reward_min: float | None = None
    reward_max: float | None = None
    pass_rate: float | None = Field(
        default=None, description="Share of scored rollouts at or above the reward threshold."
    )
    zero_variance: bool = Field(
        default=False,
        description="Every scored rollout got the same reward, so a group-relative advantage (GRPO) is zero.",
    )
    distinct_answers: int = Field(default=0, description="Distinct non-empty answers under the `AnswerFn`.")
    distinct_answer_ratio: float | None = Field(
        default=None, description="`distinct_answers / n`; low means mode collapse."
    )
    mode_answer: str | None = Field(default=None, description="The most common answer.")
    mode_answer_share: float | None = Field(default=None, description="Share of the group that gave `mode_answer`.")
    distinct_tool_sequences: int = Field(default=0, description="Distinct ordered tool-call sequences.")
    tool_usage: dict[str, int] = Field(default_factory=dict, description="Calls per tool across the group.")
    tool_calls_mean: float = 0.0
    messages_mean: float = 0.0
    lexical_diversity: float | None = Field(default=None, description="See `lexical_diversity`, over assistant text.")
    termination_reasons: dict[str, int] = Field(default_factory=dict)


def analyze_group(
    results: Sequence[RolloutResult],
    *,
    answer_fn: AnswerFn = normalized_final_response,
    reward_threshold: float = 1.0,
) -> GroupAnalysis:
    """Profile the rollouts sampled for one prompt."""
    n = len(results)
    if n == 0:
        return GroupAnalysis(n=0)

    rewards = [r.reward_result.reward for r in results if r.reward_result is not None]
    answers = [a for a in (answer_fn(r) for r in results) if a]
    answer_counts = Counter(answers)
    mode_answer, mode_count = answer_counts.most_common(1)[0] if answer_counts else (None, 0)
    sequences = [tool_sequence(r) for r in results]
    assistant_texts = [
        "\n".join(extract_message_text(m, raw=True) for m in r.messages if m.get("role") == "assistant")
        for r in results
    ]

    return GroupAnalysis(
        n=n,
        n_unscored=n - len(rewards),
        reward_mean=round(statistics.fmean(rewards), 4) if rewards else None,
        reward_std=round(statistics.pstdev(rewards), 4) if rewards else None,
        reward_min=min(rewards) if rewards else None,
        reward_max=max(rewards) if rewards else None,
        pass_rate=round(sum(r >= reward_threshold for r in rewards) / len(rewards), 4) if rewards else None,
        zero_variance=len(rewards) >= 2 and len(set(rewards)) == 1,
        distinct_answers=len(answer_counts),
        distinct_answer_ratio=round(len(answer_counts) / n, 4),
        mode_answer=mode_answer,
        mode_answer_share=round(mode_count / n, 4) if mode_answer is not None else None,
        distinct_tool_sequences=len(set(sequences)),
        tool_usage=dict(Counter(name for seq in sequences for name in seq).most_common()),
        tool_calls_mean=round(statistics.fmean(len(seq) for seq in sequences), 2),
        messages_mean=round(statistics.fmean(len(r.messages) for r in results), 2),
        lexical_diversity=lexical_diversity(assistant_texts),
        termination_reasons=dict(Counter(r.termination_reason.value for r in results).most_common()),
    )


# ---------------------------------------------------------------------------
# Run summary (across prompts)
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Group analyses averaged across prompts, each prompt weighted equally."""

    n_prompts: int
    n_samples: int
    reward_mean: float | None = None
    pass_rate: float | None = None
    zero_variance_share: float = Field(
        default=0.0, description="Share of prompts whose group carries no reward signal."
    )
    all_pass_share: float = Field(
        default=0.0, description="Zero-variance prompts that all passed: too easy to learn from."
    )
    all_fail_share: float = Field(
        default=0.0, description="Zero-variance prompts that all failed: where exploration helps."
    )
    distinct_answer_ratio_mean: float | None = None
    mode_answer_share_mean: float | None = None
    distinct_tool_sequences_mean: float = 0.0
    lexical_diversity_mean: float | None = None
    termination_reasons: dict[str, int] = Field(default_factory=dict)


def summarize_run(groups: Mapping[str, GroupAnalysis], *, reward_threshold: float = 1.0) -> RunSummary:
    """Aggregate per-prompt analyses into one run-level summary."""
    analyses = [g for g in groups.values() if g.n > 0]
    if not analyses:
        return RunSummary(n_prompts=0, n_samples=0)

    def mean_of(values: Sequence[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return round(statistics.fmean(present), 4) if present else None

    def share(predicate: Callable[[GroupAnalysis], bool]) -> float:
        return round(sum(predicate(g) for g in analyses) / len(analyses), 4)

    termination_reasons: Counter[str] = Counter()
    for g in analyses:
        termination_reasons.update(g.termination_reasons)

    return RunSummary(
        n_prompts=len(analyses),
        n_samples=sum(g.n for g in analyses),
        reward_mean=mean_of([g.reward_mean for g in analyses]),
        pass_rate=mean_of([g.pass_rate for g in analyses]),
        zero_variance_share=share(lambda g: g.zero_variance),
        all_pass_share=share(lambda g: g.zero_variance and (g.reward_min or 0.0) >= reward_threshold),
        all_fail_share=share(lambda g: g.zero_variance and (g.reward_max or 0.0) < reward_threshold),
        distinct_answer_ratio_mean=mean_of([g.distinct_answer_ratio for g in analyses]),
        mode_answer_share_mean=mean_of([g.mode_answer_share for g in analyses]),
        distinct_tool_sequences_mean=round(statistics.fmean(g.distinct_tool_sequences for g in analyses), 2),
        lexical_diversity_mean=mean_of([g.lexical_diversity for g in analyses]),
        termination_reasons=dict(termination_reasons.most_common()),
    )


# ---------------------------------------------------------------------------
# Reading eval output
# ---------------------------------------------------------------------------


def load_eval_results(path: Path | str, *, include_aborted: bool = False) -> dict[str, list[EvalSample[Any]]]:
    """Read an `Evaluator` `results.jsonl` into `{prompt_id: samples}`, without modifying the file.

    Notes:
        Unlike `Evaluator.load_results`, which rewrites the file to drop aborted rows for retry,
        this is read-only, so it is safe on a run that is still in progress.
    """
    groups: dict[str, list[EvalSample[Any]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(
                line, object_hook=lambda d: base64.b64decode(d["__bytes__"]) if set(d) == {"__bytes__"} else d
            )
            sample: EvalSample[Any] = EvalSample.model_validate(data)
            if sample.aborted and not include_aborted:
                continue
            groups.setdefault(sample.task.id.rsplit("_", 1)[0], []).append(sample)
    return groups


def format_report(summary: RunSummary, groups: Mapping[str, GroupAnalysis], *, top: int = 10) -> str:
    """Render a run summary plus the `top` least diverse prompts as a plain-text report."""

    def fmt(value: float | None, pct: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1%}" if pct else f"{value:.3g}"

    lines = [
        "─" * 72,
        f"  Rollout exploration report: {summary.n_prompts} prompts, {summary.n_samples} samples",
        "─" * 72,
        f"  reward mean              {fmt(summary.reward_mean)}",
        f"  pass rate                {fmt(summary.pass_rate, pct=True)}",
        f"  zero-variance prompts    {fmt(summary.zero_variance_share, pct=True)}"
        f"  (all pass {fmt(summary.all_pass_share, pct=True)}, all fail {fmt(summary.all_fail_share, pct=True)})",
        f"  distinct answer ratio    {fmt(summary.distinct_answer_ratio_mean)}",
        f"  mode answer share        {fmt(summary.mode_answer_share_mean, pct=True)}",
        f"  distinct tool sequences  {fmt(summary.distinct_tool_sequences_mean)}",
        f"  lexical diversity        {fmt(summary.lexical_diversity_mean)}",
        f"  terminations             {summary.termination_reasons}",
    ]
    # Least diverse first: collapsed groups are where exploration has the most to add.
    ranked = sorted(
        groups.items(),
        key=lambda item: (item[1].distinct_answer_ratio or 0.0, item[1].lexical_diversity or 0.0),
    )[:top]
    if ranked:
        lines += ["", f"  {'prompt':<28} {'n':>3} {'reward':>7} {'answers':>8} {'tool seqs':>9} {'lex div':>8}"]
        for prompt_id, g in ranked:
            lines.append(
                f"  {prompt_id[:28]:<28} {g.n:>3} {fmt(g.reward_mean):>7} {g.distinct_answers:>8}"
                f" {g.distinct_tool_sequences:>9} {fmt(g.lexical_diversity):>8}"
            )
    lines.append("─" * 72)
    return "\n".join(lines)
