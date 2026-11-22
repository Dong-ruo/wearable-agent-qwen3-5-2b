from __future__ import annotations

import copy
from datetime import timedelta

from . import SCHEMA_VERSION, TOOL_VERSION
from .contract import READ_ONLY, ToolError, canonical, error, stamp, success, tool_schemas, validate_arguments


class TaskEnvironment:
    """Independent per-episode state; reset is explicit and never reads the machine clock."""

    def __init__(self, task):
        self._task = copy.deepcopy(task)
        self.reset()

    def reset(self):
        if self._task["schema_version"] != SCHEMA_VERSION:
            raise ValueError("不支持的任务版本")
        self._state = copy.deepcopy(self._task["initial_state"])
        self._now = stamp(self._task["visible_context"]["now"])
        self._faults = copy.deepcopy(self._task["fault_plan"])
        if any(f["tool"] not in READ_ONLY or type(f["remaining"]) is not int or f["remaining"] < 0 for f in self._faults):
            raise ValueError("仅允许对只读工具注入有限次临时失败")
        simulator = self._task["user_simulator"]
        self._answers = copy.deepcopy(simulator.get("answers", {}))
        self._unanswered = set(self._answers)
        self._answered = {}
        self._followup = copy.deepcopy(simulator.get("followup"))
        self._followup_fired = False
        self._calls = 0
        self._over_budget = False
        self._protocol_errors = []
        self._done = False
        self._final = None
        self._actions = []
        self._log = []
        context = self._task["visible_context"]
        self._messages = [{"role": "system", "content": "本地穿戴助手。模拟当前时间：" + context["now"] + "，时区 Asia/Shanghai。所有工具只操作本地模拟状态。"},
                          {"role": "user", "content": context["request"]}]
        self._check_initial_state()
        return self.observation()

    def _check_initial_state(self):
        events, reminders = self._state["events"], self._state["reminders"]
        if len({e["event_id"] for e in events}) != len(events):
            raise ValueError("初始事件 ID 重复")
        if len({r["event_id"] for r in reminders}) != len(reminders):
            raise ValueError("一个事件最多一个初始提醒")
        if len({r["reminder_id"] for r in reminders}) != len(reminders):
            raise ValueError("初始提醒 ID 重复")
        ids = {e["event_id"] for e in events}
        for event in events:
            if stamp(event["end_time"]) - stamp(event["start_time"]) != timedelta(minutes=event["duration_minutes"]):
                raise ValueError("初始事件时长不一致")
        for reminder in reminders:
            if reminder["event_id"] not in ids:
                raise ValueError("初始提醒关联不存在的事件")
            event = next(e for e in events if e["event_id"] == reminder["event_id"])
            at = stamp(event["start_time"]) - timedelta(minutes=reminder["minutes_before"])
            if not self._now <= at < stamp(event["start_time"]) or at != stamp(reminder["trigger_time"]):
                raise ValueError("初始提醒时间不一致或落在过去")

    def observation(self):
        """Explicit whitelist. No complete task/state, goals, labels, plans or unasked answers."""
        return {"messages": copy.deepcopy(self._messages), "tools": tool_schemas(self._task.get("tool_order")),
                "current_time": self._task["visible_context"]["now"], "timezone": "Asia/Shanghai"}

    def private_snapshot(self):
        return copy.deepcopy({"state": self._state, "faults": self._faults, "answered": self._answered,
            "unanswered": sorted(self._unanswered), "followup_fired": self._followup_fired,
            "user_answers": self._answers, "followup": self._followup, "call_log": self._log,
            "tool_calls": self._calls, "over_budget": self._over_budget, "done": self._done, "final": self._final})

    def _event(self, event_id):
        value = next((e for e in self._state["events"] if e["event_id"] == event_id), None)
        if value is None:
            raise ToolError("NOT_FOUND", event_id=event_id)
        return copy.deepcopy(value)

    def _activity(self, activity_id):
        value = next((a for a in self._state["activities"] if a["activity_id"] == activity_id), None)
        if value is None:
            raise ToolError("NOT_FOUND", activity_id=activity_id)
        return copy.deepcopy(value)

    def _conflicts(self, start, end, exclude=None):
        return sorted(e["event_id"] for e in self._state["events"] if e["event_id"] != exclude and
                      stamp(e["start_time"]) < end and stamp(e["end_time"]) > start)

    def _validate_window(self, start_time, duration, exclude=None):
        start = stamp(start_time)
        if start < self._now:
            raise ToolError("INVALID_ARGUMENT", reason="活动不能安排在过去")
        end = start + timedelta(minutes=duration)
        conflicts = self._conflicts(start, end, exclude)
        if conflicts:
            raise ToolError("TIME_CONFLICT", conflicting_event_ids=conflicts)
        return start, end

    def _allocate(self, key, prefix, existing):
        n = self._state[key]
        while f"{prefix}_{n:05d}" in existing:
            n += 1
        self._state[key] = n + 1
        return f"{prefix}_{n:05d}"

    def _invoke(self, name, args):
        validate_arguments(name, args)
        for rule in self._faults:
            if rule["tool"] == name and rule["remaining"] and all(args.get(k) == v for k, v in rule.get("match", {}).items()):
                rule["remaining"] -= 1
                raise ToolError("TEMPORARY_UNAVAILABLE")
        return getattr(self, "_tool_" + name)(args)

    def execute(self, name, arguments):
        """All calls, including reference-policy calls, pass through step and consume budget."""
        return self.step({"type": "tool_call", "name": name, "arguments": arguments})["observation"]["tool_result"]

    def _tool_list_events(self, args):
        start, end = stamp(args["start_time"]), stamp(args["end_time"])
        rows = [copy.deepcopy(e) for e in self._state["events"] if stamp(e["start_time"]) < end and stamp(e["end_time"]) > start]
        return {"events": sorted(rows, key=lambda e: (e["start_time"], e["event_id"]))}

    def _tool_create_event(self, args):
        activity = self._activity(args["activity_id"])
        start, end = self._validate_window(args["start_time"], args["duration_minutes"])
        event_id = self._allocate("next_event_id", "event", {e["event_id"] for e in self._state["events"]})
        event = {"event_id": event_id, "activity_id": args["activity_id"], "title": activity["title"],
                 "start_time": start.isoformat(), "end_time": end.isoformat(), "duration_minutes": args["duration_minutes"]}
        self._state["events"].append(event)
        return copy.deepcopy(event)

    def _tool_update_event(self, args):
        event = self._event(args["event_id"])
        if "activity_id" in args:
            activity = self._activity(args["activity_id"])
            event.update(activity_id=activity["activity_id"], title=activity["title"])
        start, end = self._validate_window(args.get("start_time", event["start_time"]), args.get("duration_minutes", event["duration_minutes"]), event["event_id"])
        event.update(start_time=start.isoformat(), end_time=end.isoformat(), duration_minutes=int((end-start).total_seconds()/60))
        reminders = copy.deepcopy(self._state["reminders"])
        for reminder in reminders:
            if reminder["event_id"] == event["event_id"]:
                at = start - timedelta(minutes=reminder["minutes_before"])
                if not self._now <= at < start:
                    raise ToolError("INVALID_ARGUMENT", reason="改时后的提醒在过去")
                reminder["trigger_time"] = at.isoformat()
        self._state["events"] = [event if e["event_id"] == event["event_id"] else e for e in self._state["events"]]
        self._state["reminders"] = reminders
        return copy.deepcopy(event)

    def _tool_delete_event(self, args):
        event = self._event(args["event_id"])
        removed = sorted(r["reminder_id"] for r in self._state["reminders"] if r["event_id"] == event["event_id"])
        self._state["events"] = [e for e in self._state["events"] if e["event_id"] != event["event_id"]]
        self._state["reminders"] = [r for r in self._state["reminders"] if r["event_id"] != event["event_id"]]
        return {"event_id": event["event_id"], "deleted": True, "deleted_reminder_ids": removed}

    def _tool_get_weather(self, args):
        condition = self._state["weather"].get(args["city"] + "|" + args["date"])
        if condition is None:
            raise ToolError("NOT_FOUND", city=args["city"], date=args["date"])
        return {"city": args["city"], "date": args["date"], "condition": condition}

    def _tool_search_activity(self, args):
        rows = [copy.deepcopy(a) for a in self._state["activities"] if
                ("activity_id" not in args or a["activity_id"] == args["activity_id"]) and
                ("indoor" not in args or a["indoor"] == args["indoor"]) and
                ("max_duration_minutes" not in args or a["duration_minutes"] <= args["max_duration_minutes"])]
        return {"activities": sorted(rows, key=lambda a: a["activity_id"])}

    def _tool_set_reminder(self, args):
        event = self._event(args["event_id"])
        at = stamp(event["start_time"]) - timedelta(minutes=args["minutes_before"])
        if not self._now <= at < stamp(event["start_time"]):
            raise ToolError("INVALID_ARGUMENT", reason="提醒必须早于活动且不早于模拟当前时间")
        old = next((r for r in self._state["reminders"] if r["event_id"] == event["event_id"]), None)
        if old:
            reminder_id = old["reminder_id"]
        else:
            reminder_id = self._allocate("next_reminder_id", "reminder", {r["reminder_id"] for r in self._state["reminders"]})
        reminder = {"reminder_id": reminder_id, "event_id": event["event_id"], "minutes_before": args["minutes_before"], "trigger_time": at.isoformat()}
        self._state["reminders"] = [r for r in self._state["reminders"] if r["event_id"] != event["event_id"]] + [reminder]
        return copy.deepcopy(reminder)

    def _tool_get_preferences(self, args):
        return {"preferences": copy.deepcopy(self._state["preferences"])}

    def step(self, action):
        if self._done:
            raise ValueError("任务已经结束，请 reset 后重放")
        action = copy.deepcopy(action)
        if not isinstance(action, dict):
            self._protocol_errors.append("INVALID_ACTION_OBJECT")
            self._done = True
            self._actions.append({"action": action, "observation": {"messages": []}})
            return {"observation": {"messages": []}, "done": True, "reward": 0}
        before = len(self._messages)
        tool_result = None
        user_reply = None
        kind = action.get("type")
        if kind == "tool_call":
            self._calls += 1
            name, args = action.get("name"), action.get("arguments")
            malformed = not isinstance(name, str) or not isinstance(args, dict)
            if malformed:
                self._protocol_errors.append("INVALID_TOOL_CALL")
                name, args = "__invalid__", {}
            call_id = f"call_{self._calls:04d}"
            self._messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": copy.deepcopy(args)}}]})
            if self._calls > self._task["max_tool_calls"]:
                self._over_budget = True
                tool_result = error("BUDGET_EXCEEDED")
                self._done = True
            elif malformed:
                tool_result = error("INVALID_ARGUMENT", reason="tool_call 需要名称字符串及参数对象")
            else:
                try:
                    tool_result = success(self._invoke(name, args))
                except (ValueError, TypeError, KeyError) as exc:
                    tool_result = error("INVALID_ARGUMENT", reason=str(exc))
                except ToolError as exc:
                    tool_result = error(exc.code, **exc.data)
            self._messages.append({"role": "tool", "name": name, "tool_call_id": call_id, "content": canonical(tool_result)})
            self._log.append({"index": len(self._log), "name": name, "arguments": copy.deepcopy(args), "result": copy.deepcopy(tool_result), "action_index": len(self._actions)})
            if tool_result["status"] == "success" and name == "create_event" and self._followup and not self._followup_fired:
                if self._followup.get("trigger") != "after_create_success":
                    raise ValueError("未知后续用户消息触发规则")
                self._followup_fired = True
                user_reply = copy.deepcopy(self._followup["reply"])
                self._messages.append({"role": "user", "content": user_reply["content"]})
        elif kind == "ask_user":
            slots = action.get("requested_slots", [])
            question = action.get("question", "")
            if not isinstance(slots, list) or not slots or not all(isinstance(s, str) for s in slots) or len(slots) != len(set(slots)) or not set(slots) <= self._unanswered or not isinstance(question, str) or not question.strip():
                self._protocol_errors.append("INVALID_CLARIFICATION")
                user_reply = {"error_code": "NO_RELEVANT_MISSING_SLOT"}
                self._done = True
            else:
                values = {slot: copy.deepcopy(self._answers[slot]) for slot in slots}
                self._unanswered -= set(slots)
                self._answered.update(values)
                user_reply = {"slots": values, "content": "；".join(k + "=" + str(values[k]) for k in slots)}
                self._messages.extend([{"role": "assistant", "content": question}, {"role": "user", "content": user_reply["content"]}])
        elif kind == "final":
            if not isinstance(action.get("status"), str) or action.get("status") not in {"success", "failed"} or not isinstance(action.get("result"), dict) or not isinstance(action.get("message"), str):
                self._protocol_errors.append("INVALID_FINAL")
            self._final = action
            self._messages.append({"role": "assistant", "content": canonical({k: action.get(k) for k in ("status", "message", "result")})})
            self._done = True
        else:
            self._protocol_errors.append("UNKNOWN_ACTION")
            self._done = True
        delta = {"messages": copy.deepcopy(self._messages[before:])}
        if tool_result is not None:
            delta["tool_result"] = tool_result
        if user_reply is not None:
            delta["user_reply"] = user_reply
        self._actions.append({"action": action, "observation": copy.deepcopy(delta)})
        reward = self.evaluate()["reward"] if self._done else 0
        return {"observation": delta, "done": self._done, "reward": reward}

    def evaluate(self):
        from .validation import validate
        return validate(self)

    def trajectory(self):
        return {"schema_version": SCHEMA_VERSION, "tool_version": TOOL_VERSION, "case_id": self._task["case_id"],
                "actions": copy.deepcopy(self._actions), "messages": copy.deepcopy(self._messages),
                "final_state": copy.deepcopy(self._state), "validation": self.evaluate()}
