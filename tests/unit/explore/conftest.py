from strands_env.core import RewardResult, RolloutResult, TerminationReason


def make_result(
    answer: str = "42",
    reward: float | None = 1.0,
    tools: tuple[str, ...] = (),
    termination: TerminationReason = TerminationReason.TASK_COMPLETE,
) -> RolloutResult:
    """Build a rollout that calls `tools` in order, then answers `answer`."""
    messages = []
    for i, name in enumerate(tools):
        messages.append(
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": f"t{i}", "name": name, "input": {"x": i}}}]}
        )
        messages.append(
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": f"t{i}", "status": "success", "content": [{"text": "ok"}]}}],
            }
        )
    messages.append({"role": "assistant", "content": [{"text": f"The answer is {answer}"}]})
    return RolloutResult(
        messages=messages,
        reward_result=RewardResult(reward=reward) if reward is not None else None,
        termination_reason=termination,
    )
