"""Terminal binary reward from state, authorized effects, and semantically required evidence."""
from __future__ import annotations

import copy
from datetime import timedelta

from .contract import WRITES, canonical, stamp


def validate(env):
    goal, state = env._task["goal"], env._state
    logs, actions = env._log, env._actions
    good = [r for r in logs if r["result"]["status"] == "success"]
    created = [r for r in good if r["name"] == "create_event"]
    created_id = created[0]["result"]["data"]["event_id"] if created else None

    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return created_id if value == "$created" else value

    expected_events = resolve(goal["expected_events"])
    expected_reminders = resolve(goal["expected_reminders"])
    checks = {"NORMAL_TERMINATION": env._done and env._final is not None and not env._protocol_errors,
              "WITHIN_BUDGET": not env._over_budget and env._calls <= env._task["max_tool_calls"],
              "SUCCESS_FEEDBACK": bool(env._final and env._final.get("status") == "success")}
    checks["EVENTS_EXACT"] = sorted(state["events"], key=lambda e: e["event_id"]) == sorted(expected_events, key=lambda e: str(e["event_id"]))
    checks["REMINDERS_EXACT"] = len(state["reminders"]) == len(expected_reminders) and all(
        sum(all(actual.get(k) == v for k, v in expected.items()) for actual in state["reminders"]) == 1 for expected in expected_reminders)
    checks["REMINDER_TIMES_VALID"] = all(any(e["event_id"] == r["event_id"] and
        stamp(r["trigger_time"]) == stamp(e["start_time"])-timedelta(minutes=r["minutes_before"]) and
        env._now <= stamp(r["trigger_time"]) < stamp(e["start_time"]) for e in state["events"]) for r in state["reminders"])
    # Only task-authorized writes may succeed, even if an attacker later repairs the final state.
    writes_allowed = len(created) == (1 if goal["create_spec"] else 0)
    update_id, delete_id, reminder_id = (resolve(goal[k]) for k in ("update_event_id", "delete_event_id", "reminder_target"))
    for row in good:
        name, args, data = row["name"], row["arguments"], row["result"]["data"]
        if name == "create_event":
            writes_allowed &= all(data.get(k) == v for k, v in goal["create_spec"].items()) if goal["create_spec"] else False
        elif name == "update_event":
            target = next((e for e in expected_events if e["event_id"] == update_id), None)
            writes_allowed &= args["event_id"] == update_id and target is not None and data == target
        elif name == "delete_event":
            writes_allowed &= args["event_id"] == delete_id
        elif name == "set_reminder":
            matching = next((r for r in expected_reminders if r["event_id"] == reminder_id), None)
            writes_allowed &= args["event_id"] == reminder_id and matching is not None and args["minutes_before"] == matching["minutes_before"]
    checks["AUTHORIZED_WRITES"] = bool(writes_allowed)
    checks["REQUIRED_UPDATE"] = update_id is None or any(r["name"] == "update_event" and r["arguments"]["event_id"] == update_id for r in good)
    checks["REQUIRED_DELETE"] = delete_id is None or any(r["name"] == "delete_event" and r["arguments"]["event_id"] == delete_id for r in good)
    checks["REQUIRED_REMINDER"] = reminder_id is None or any(r["name"] == "set_reminder" and r["arguments"]["event_id"] == reminder_id for r in good)
    if goal["kind"] == "conflict_fallback":
        intent = env._task["reference_intent"]
        allowed = [intent["start_time"]]+intent["alternatives"]
        chosen = goal["create_spec"]["start_time"]
        previous = allowed[:allowed.index(chosen)]
        created_at = created[0]["action_index"] if created else -1
        checks["CONFLICT_DISCOVERED"] = all(any(r["name"] == "create_event" and r["arguments"].get("start_time") == when and
            r["result"]["status"] == "error" and r["result"]["error_code"] == "TIME_CONFLICT" and r["action_index"] < created_at for r in logs) for when in previous)
    first_write = min((r["action_index"] for r in logs if r["name"] in WRITES), default=len(actions))
    for i, requirement in enumerate(goal["required_queries"]):
        matches = []
        for row in good:
            if row["name"] != requirement["name"] or row["action_index"] >= first_write:
                continue
            if "arguments" in requirement and row["arguments"] != requirement["arguments"]:
                continue
            if "contains_event_id" in requirement and not any(e["event_id"] == requirement["contains_event_id"] for e in row["result"]["data"]["events"]):
                continue
            if requirement.get("selected_activity") and not any(a["activity_id"] == goal["create_spec"]["activity_id"] for a in row["result"]["data"]["activities"]):
                continue
            matches.append(row)
        checks[f"NECESSARY_QUERY_{i}"] = bool(matches)
    for slot in goal["required_asks"]:
        checks["ASKED_"+slot] = any(i < first_write and isinstance(row["action"], dict) and row["action"].get("type") == "ask_user" and
            slot in row["observation"].get("user_reply", {}).get("slots", {}) and
            row["observation"]["user_reply"]["slots"][slot] == env._task["user_simulator"]["answers"][slot]
            for i, row in enumerate(actions))
    if goal["requires_followup"]:
        creation_index = created[0]["action_index"] if created else len(actions)
        later = [r for r in good if r["action_index"] > creation_index and r["name"] in {"update_event", "delete_event"}]
        checks["FOLLOWUP_HANDLED"] = env._followup_fired and bool(later)
        if goal["kind"] == "weather_reschedule_remind":
            updates = [r["action_index"] for r in later if r["name"] == "update_event"]
            checks["COMPOSITION_ORDER"] = bool(updates) and any(r["name"] == "set_reminder" and r["action_index"] > min(updates) for r in good)
    final_result = env._final.get("result", {}) if env._final else {}
    if not isinstance(final_result, dict):
        final_result = {}
    if goal["query"]:
        query = goal["query"]
        matching = [r for r in good if r["name"] == query["name"] and r["arguments"] == query["arguments"]]
        checks["QUERY_FACTS"] = bool(matching) and final_result == matching[-1]["result"]["data"]
    else:
        intended_id = created_id or update_id or delete_id or reminder_id
        checks["FINAL_OBJECT"] = intended_id is not None and final_result.get("event_id") == intended_id
        if "reminder_id" in final_result:
            checks["FINAL_REMINDER_OBJECT"] = any(r["reminder_id"] == final_result["reminder_id"] and r["event_id"] == intended_id for r in state["reminders"])
    passed = all(checks.values())
    return {"success": passed, "reward": int(passed), "checks": checks,
            "failure_codes": sorted(k for k, v in checks.items() if not v),
            "feedback_scope": "结构化 final.result 与事实一致性；自然语言说明的完整忠实度未自动验证"}
