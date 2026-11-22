"""Independent assertions over final state and trace, rather than tool-JSON matching."""
from __future__ import annotations

import statistics
from collections import Counter
from datetime import timedelta

from .cases import Case
from .environment import dt
from .runtime import Runtime
from .rewards import cost_metrics


def assess(case: Case, runtime: Runtime) -> dict:
    state, trace = runtime.env.snapshot(), runtime.env.trace
    final = runtime.last_final
    failures = []
    events = [e for e in state["events"] if e.get("origin") == "agent"]
    reminders = state["reminders"]
    if final is None or final.status != case.expected_status:
        failures.append("wrong_outcome")
    expected_errors = {
        "event_failure": ("calendar_create", "write_failed"),
        "reminder_failure": ("reminder_create", "write_failed"),
        "verify_failure": ("calendar_get", "unavailable"),
        "reminder_verify_failure": ("reminder_get", "unavailable"),
        "weather_missing": ("weather_get", "weather_missing"),
        "calendar_unavailable": ("calendar_list", "unavailable"),
        "no_slot": ("calendar_find_slot", "no_slot"),
    }
    if case.name in expected_errors:
        name, code = expected_errors[case.name]
        if not any(r["name"] == name and not r["result"]["ok"] and r["result"]["error"]["code"] == code for r in trace):
            failures.append("missing_expected_failure_evidence")
    expected_slots = {"missing_city": "city", "ambiguous_time": "expression", "busy": "confirm_time", "race": "confirm_time"}
    if case.name in expected_slots and (final is None or expected_slots[case.name] not in final.requested_slots):
        failures.append("wrong_clarification")
    if len(events) != case.expected_event_count or len(reminders) != case.expected_reminder_count:
        failures.append("wrong_write_count")
    for original in case.scenario.events:
        if original not in state["events"]:
            failures.append("existing_event_changed")
    for event in events:
        if event.get("activity") != case.expected_activity or event.get("city") != case.city:
            failures.append("wrong_branch_or_city")
        if dt(event["start"]) != dt(case.expected_start):
            failures.append("wrong_time")
        if dt(event["end"]) - dt(event["start"]) != timedelta(minutes=case.duration_minutes):
            failures.append("wrong_duration")
        if any(other["id"] != event["id"] and dt(other["start"]) < dt(event["end"]) and dt(other["end"]) > dt(event["start"])
               for other in state["events"]):
            failures.append("calendar_overlap")
        created = next((r for r in trace if r["name"] == "calendar_create" and r["result"]["ok"] and r["result"]["data"]["id"] == event["id"]), None)
        if not created:
            failures.append("missing_create_evidence")
        else:
            before = trace[:created["index"]]
            if not any(r["name"] == "calendar_list" and r["result"]["ok"] and
                       dt(r["arguments"]["start"]) == dt(event["start"]) and
                       dt(r["arguments"]["end"]) >= dt(event["end"]) and
                       not r["result"]["data"]["events"] for r in before):
                failures.append("missing_availability_check")
            if not any(r["name"] == "weather_get" and r["result"]["ok"] and
                       r["arguments"]["city"] == case.city and r["arguments"]["date"] == dt(event["start"]).date().isoformat() for r in before):
                failures.append("missing_weather_check")
    for reminder in reminders:
        event = next((e for e in events if e["id"] == reminder["event_id"]), None)
        if event is None or reminder["minutes_before"] != case.reminder_minutes or dt(reminder["remind_at"]) != dt(event["start"]) - timedelta(minutes=case.reminder_minutes):
            failures.append("wrong_reminder")
    if case.requires_consent and events:
        # Check a clarification happened before the second user turn and before any successful write.
        history = runtime.messages
        user_positions = [i for i, m in enumerate(history) if m["role"] == "user"]
        if len(user_positions) < 2 or not runtime.finals or runtime.finals[0]["status"] != "clarification":
            failures.append("missing_consent")
        elif any(m["role"] == "tool" and '"origin": "agent"' in m["content"] for m in history[:user_positions[1]]):
            failures.append("write_before_consent")
    claimed_success = bool(final and final.status == "success")
    raw = runtime.raw_finals[-1] if runtime.raw_finals else None
    if raw and raw["status"] == "success" and not claimed_success:
        failures.append("unsupported_success_claim")
    goal_met = (len(events) == 1 and len(reminders) == 1 and not failures and claimed_success)
    return {"case": case.name, "passed": not failures, "task_completed": goal_met,
            "status": final.status if final else None, "failures": sorted(set(failures)),
            "tool_calls": len(trace), "tool_errors": sum(not r["result"]["ok"] for r in trace),
            "elapsed_seconds": round(runtime.elapsed_seconds, 4),
            "model_http_seconds": runtime.policy_seconds if hasattr(runtime.policy, "request_count") else None,
            "policy_seconds": runtime.policy_seconds, "tool_seconds": runtime.tool_seconds,
            "step_limit_reached": runtime.step_limit_reached,
            "recoverable_fault_case": bool(case.scenario.faults) and case.expected_status == "success",
            "clarification_case": case.name in {"city_followup", "time_followup"},
            "parameter_semantic_correct": None if not events else not any(f in failures for f in ("wrong_branch_or_city", "wrong_time", "wrong_duration", "wrong_reminder")),
            "generated_tokens": (sum(u["completion_tokens"] for u in runtime.policy.usage) if getattr(runtime.policy, "usage", None) and len(runtime.policy.usage) == runtime.policy.request_count and all("completion_tokens" in u for u in runtime.policy.usage) else None),
            "ttft_seconds": None, "peak_model_memory_bytes": None,
            "tool_selection_accuracy": None,
            **cost_metrics(trace)}


def aggregate(rows: list[dict], backend: str) -> dict:
    failure_counts = Counter(f for row in rows for f in row["failures"])
    def conditional_rate(flag):
        subset = [r for r in rows if r.get(flag)]
        return sum(r["task_completed"] for r in subset) / len(subset) if subset else None
    return {"backend": backend, "case_count": len(rows),
            "outcome_accuracy": sum(r["passed"] for r in rows) / len(rows) if rows else 0,
            "task_completion_rate": sum(r["task_completed"] for r in rows) / len(rows) if rows else 0,
            "mean_tool_calls": statistics.mean(r["tool_calls"] for r in rows) if rows else 0,
            "recovery_success_rate": conditional_rate("recoverable_fault_case"),
            "clarification_completion_rate": conditional_rate("clarification_case"),
            "step_limit_rate": sum(r.get("step_limit_reached", False) for r in rows) / len(rows) if rows else 0,
            "mean_redundant_calls": statistics.mean(r.get("redundant_calls", 0) for r in rows) if rows else 0,
            "failure_counts": dict(failure_counts), "results": rows,
            "note": "固定工程回归场景；澄清和正确报告失败计入 outcome_accuracy，不计入 task_completion_rate。规则基线成绩不代表模型能力。"}

