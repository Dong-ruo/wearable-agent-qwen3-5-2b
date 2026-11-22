"""Versioned deterministic rubric and bounded reward; no LLM judge or task generator."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

from .environment import Environment
from .goals import GoalSpec, evaluate_goal
from .models import Final


@dataclass(frozen=True)
class RewardConfig:
    process_weight: float = 0.1
    cost_weight: float = 0.05
    mode: str = "result_process"

    def __post_init__(self):
        if not (0 <= self.process_weight <= 0.2 and 0 <= self.cost_weight <= 0.2):
            raise ValueError("辅助奖励与成本权重各限制在 [0,0.2]，保证成功奖励占主导")
        if self.mode not in {"result_only", "result_process"}:
            raise ValueError("未知奖励模式")


def cost_metrics(trace: list[dict]) -> dict:
    invalid = sum(not r["result"]["ok"] and r["result"]["error"]["code"] in {"unknown_tool", "invalid_arguments", "invalid_value"} for r in trace)
    redundant = 0
    for before, after in zip(trace, trace[1:]):
        if before["name"] not in {"calendar_list", "calendar_get", "reminder_get", "weather_get", "preferences_get", "activity_search"}:
            continue
        redundant += (before["result"]["ok"] and after["result"]["ok"] and before["name"] == after["name"] and
                      before["arguments"] == after["arguments"] and before["result"] == after["result"])
    return {"invalid_calls": invalid, "redundant_calls": redundant, "cost": min(1.0, (invalid + redundant) / 4)}


def score_episode(goal: GoalSpec, env: Environment, final: Final | None, config: RewardConfig = RewardConfig(),
                  trace_start=0, initial_state=None) -> dict:
    checks = evaluate_goal(goal, env, trace_start, initial_state)
    state_success = all(checks.values())
    truthful = final is not None and (final.status != "success" or state_success)
    if final:
        if final.event_id:
            truthful &= any(e["id"] == final.event_id for e in env.events) or final.event_id in goal.deleted_event_ids
        if final.reminder_id:
            truthful &= any(r["id"] == final.reminder_id for r in env.reminders)
    success = state_success and truthful and final is not None and final.status == "success"
    checks["structured_feedback_consistent"] = bool(truthful)
    process = sum(checks.values()) / len(checks)
    costs = cost_metrics(env.trace[trace_start:])
    total = float(success)
    if config.mode == "result_process":
        total += config.process_weight * process - config.cost_weight * costs["cost"]
    return {"rubric_version": "manual-v1", "reward": total, "success": bool(success),
            "state_success": state_success, "process_score": process, **costs, "checks": checks,
            "feedback_scope": "仅验证结构化状态与对象ID，不对自由文本解释进行语义裁判"}


def group_statistics(rewards: list[float]) -> dict:
    if not rewards or any(not math.isfinite(r) for r in rewards):
        raise ValueError("奖励组必须非空且数值有限")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return {"mean": mean, "std": std, "constant_group": std < 1e-8,
            "advantages": [(r - mean) / (std + 1e-8) for r in rewards]}


def trl_reward(completions, environments, **kwargs):
    """The only reward source passed to TRL; the adapter intentionally has no get_reward."""
    if len(completions) != len(environments):
        raise ValueError("completion 与环境数量不匹配")
    results = []
    for completion, wrapper in zip(completions, environments):
        text = completion if isinstance(completion, str) else (completion[-1].get("content") or "")
        try:
            final = Final.model_validate_json(text)
        except (ValueError, TypeError):
            final = None
        report = score_episode(wrapper._goal, wrapper._env, final, wrapper._reward_config)
        wrapper._last_reward = report
        results.append(report["reward"])
    return results


def recorded_trl_reward(output: Path, group_size: int):
    """Persist auditable per-rollout rewards and group diagnostics; never add a second reward."""
    output.mkdir(parents=True, exist_ok=True)

    def rubric_reward(completions, environments, **kwargs):
        rewards = trl_reward(completions, environments, **kwargs)
        groups = defaultdict(list)
        with (output / "rollout-rewards.jsonl").open("a", encoding="utf-8") as handle:
            for wrapper, reward in zip(environments, rewards):
                key = json.dumps([wrapper._scenario.model_dump(mode="json"), wrapper._request, wrapper._goal.model_dump(mode="json")], sort_keys=True, ensure_ascii=False)
                groups[key].append(wrapper._last_reward)
                handle.write(json.dumps({"reward_report": wrapper._last_reward, "trace": wrapper._env.trace,
                                         "state": wrapper._env.snapshot()}, ensure_ascii=False) + "\n")
        diagnostics = []
        for rows in groups.values():
            for offset in range(0, len(rows), group_size):
                group = rows[offset:offset+group_size]
                diagnostics.append({**group_statistics([r["reward"] for r in group]),
                    "size": len(group), "complete_group": len(group) == group_size,
                    "all_failure": all(not r["success"] for r in group), "all_success": all(r["success"] for r in group),
                    "invalid_calls": sum(r["invalid_calls"] for r in group), "redundant_calls": sum(r["redundant_calls"] for r in group)})
        with (output / "group-diagnostics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostics, ensure_ascii=False) + "\n")
        return rewards
    return rubric_reward
