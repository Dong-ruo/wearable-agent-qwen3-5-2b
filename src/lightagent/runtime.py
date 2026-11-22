from __future__ import annotations

import copy
import json
import time
from datetime import timedelta

from .environment import Environment, dt, tool_specs
from .models import Call, Final
from .goals import GoalSpec, evaluate_goal
from .policy import Policy, SYSTEM


class Runtime:
    def __init__(self, env: Environment, policy: Policy, max_steps: int = 20, goal: GoalSpec | None = None):
        self.env, self.policy, self.max_steps = env, policy, max_steps
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.finals: list[dict] = []
        self.elapsed_seconds = 0.0
        self.last_final: Final | None = None
        self.on_tool = None
        self.start_trace_index = len(env.trace)
        self.initial_event_ids = {e["id"] for e in env.events}
        self.initial_reminder_ids = {r["id"] for r in env.reminders}
        self.initial_state = env.snapshot()
        self.goal = goal
        self.raw_finals: list[dict] = []
        self.policy_seconds = 0.0
        self.tool_seconds = 0.0
        self.step_limit_reached = False

    def task_events(self):
        return [e for e in self.env.events if e.get("origin") == "agent" and e["id"] not in self.initial_event_ids]

    def task_reminders(self):
        return [r for r in self.env.reminders if r["id"] not in self.initial_reminder_ids]

    def task_mutations(self):
        return [r for r in self.env.trace[self.start_trace_index:] if r["result"]["ok"] and r["name"] in
                {"calendar_create", "calendar_update", "calendar_delete", "reminder_create"}]

    def _last_error(self):
        for row in reversed(self.env.trace[self.start_trace_index:]):
            if not row["result"]["ok"]:
                return " 工具反馈：" + row["result"]["error"]["message"] + "。"
        return ""

    def run(self, text: str) -> Final:
        if self.last_final and self.last_final.status != "clarification":
            raise ValueError("当前任务已结束，请为新任务创建新的 Runtime")
        self.messages.append({"role": "user", "content": text})
        started = time.monotonic()
        final = None
        for _ in range(self.max_steps):
            tick = time.monotonic()
            try:
                action = self.policy.next(copy.deepcopy(self.messages), tool_specs())
            except (RuntimeError, ValueError, TypeError) as exc:
                final = self._interrupted(f"策略调用失败：{exc}。")
                break
            finally:
                self.policy_seconds += time.monotonic() - tick
            if isinstance(action, Final):
                self.raw_finals.append(action.model_dump())
                final = self._ground(action)
                break
            if not isinstance(action, Call):
                final = self._interrupted("策略返回了不支持的动作。")
                break
            call_id = f"call_{len(self.env.trace)}"
            self.messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": action.name, "arguments": json.dumps(action.arguments, ensure_ascii=False)}}]})
            tick = time.monotonic()
            observation = self.env.execute(action.name, action.arguments)
            self.tool_seconds += time.monotonic() - tick
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": json.dumps(observation, ensure_ascii=False)})
            if self.on_tool:
                self.on_tool(self.env.trace[-1])
        if final is None:
            self.step_limit_reached = True
            final = self._interrupted(f"已达到每轮 {self.max_steps} 步上限，任务停止。")
        self.elapsed_seconds += time.monotonic() - started
        self.messages.append({"role": "assistant", "content": final.model_dump_json()})
        self.finals.append(final.model_dump())
        self.last_final = final
        return final

    def _interrupted(self, message):
        events = self.task_events()
        if events:
            return Final(status="partial", message=message + "已有日程写入，尚未确认全部任务完成。", event_id=events[-1]["id"])
        reminders = self.task_reminders()
        if reminders:
            return Final(status="partial", message=message + "已有提醒写入，尚未确认全部任务完成。",
                         event_id=reminders[-1]["event_id"], reminder_id=reminders[-1]["id"])
        if self.task_mutations():
            return Final(status="partial", message=message + "本次已有修改或删除操作生效，请检查实际状态。")
        return Final(status="failed", message=message + "未创建日程或提醒。")

    def _evidence(self, name, object_id):
        for row in reversed(self.env.trace[self.start_trace_index:]):
            if row["name"] == name and row["result"]["ok"] and row["result"]["data"].get("id") == object_id:
                return row
        return None

    def _ground(self, final: Final) -> Final:
        """Render success from persisted state, never from a model's unsupported claim."""
        if self.goal is not None and final.status == "success":
            checks = evaluate_goal(self.goal, self.env, self.start_trace_index, self.initial_state)
            ids_exist = (not final.event_id or any(e["id"] == final.event_id for e in self.env.events) or final.event_id in self.goal.deleted_event_ids) and (not final.reminder_id or any(r["id"] == final.reminder_id for r in self.env.reminders))
            if not all(checks.values()) or not ids_exist:
                return self._interrupted("目标状态或核验证据不满足要求，已拦截成功反馈。")
            facts = [r["result"]["data"] for r in self.env.trace[self.start_trace_index:] if r["name"] == "weather_get" and r["result"]["ok"]]
            final.message = "本地模拟任务的目标状态与核验检查通过。" + (" 查询结果：" + json.dumps(facts, ensure_ascii=False) if facts else "")
            return final
        if final.status == "success":
            if len(self.task_events()) != 1 or len(self.task_reminders()) != 1:
                return self._interrupted("本次写入数量与单项活动任务不符，已拦截成功反馈。")
            event = next((e for e in self.task_events() if e["id"] == final.event_id), None)
            reminder = next((r for r in self.env.reminders if r["id"] == final.reminder_id and r["event_id"] == final.event_id), None)
            created = self._evidence("calendar_create", final.event_id)
            verified = self._evidence("calendar_get", final.event_id)
            reminded = self._evidence("reminder_create", final.reminder_id)
            checked = self._evidence("reminder_get", final.reminder_id)
            if not all((event, reminder, created, verified, reminded, checked)):
                return self._interrupted("策略声称成功，但缺少日程/提醒创建或回读证据，已拦截成功反馈。")
            if verified["index"] <= created["index"] or checked["index"] <= reminded["index"]:
                return self._interrupted("回读发生在写入之前，不能确认当前状态。")
            if verified["result"]["data"] != event or checked["result"]["data"] != reminder:
                return self._interrupted("回读结果与当前状态不一致，不能确认完成。")
            if dt(reminder["remind_at"]) != dt(event["start"]) - timedelta(minutes=reminder["minutes_before"]):
                return self._interrupted("提醒时间不一致，不能确认完成。")
            start, end = dt(event["start"]), dt(event["end"])
            final.message = (f"已在本地模拟环境创建并核验：{start:%Y-%m-%d %H:%M}–{end:%H:%M} "
                             f"{event['city']}{event['title']}（{self.env.scenario.timezone}）；"
                             f"提醒时间 {dt(reminder['remind_at']):%Y-%m-%d %H:%M}，提前 {reminder['minutes_before']} 分钟。"
                             "这是模拟提醒，不会实际推送。")
            return final
        # For unsuccessful outcomes, disclose persisted writes even if a policy omits them.
        events = self.task_events()
        if events:
            event = next((e for e in events if e["id"] == final.event_id), events[-1])
            reminders = [r for r in self.env.reminders if r["event_id"] == event["id"]]
            event_checked = bool(self._evidence("calendar_get", event["id"]))
            reminder_checked = bool(reminders and self._evidence("reminder_get", reminders[-1]["id"]))
            # Keep clarification possible if a model asks how to repair a partial write.
            status = "clarification" if final.status == "clarification" else "partial"
            detail = (f"实际状态：日程 {event['id']} 已写入（{'已回读' if event_checked else '未完成回读'}）；"
                      f"提醒{'已写入并回读' if reminder_checked else '已写入但未完成回读' if reminders else '未创建'}。")
            explanation = final.message if status == "clarification" else "未确认全部任务完成。" + self._last_error()
            return Final(status=status, message=detail + " " + explanation, event_id=event["id"],
                         reminder_id=reminders[-1]["id"] if reminders else None, requested_slots=final.requested_slots)
        new_reminders = self.task_reminders()
        if new_reminders:
            return Final(status="partial", message="实际状态：本次未创建日程，但已为已有日程写入提醒，尚未确认目标完成。",
                         event_id=new_reminders[-1]["event_id"], reminder_id=new_reminders[-1]["id"])
        if self.task_mutations():
            return Final(status="partial", message="本次已有修改或删除操作生效，但尚未确认目标完成。" + self._last_error())
        if final.status == "partial" or final.event_id or final.reminder_id:
            return Final(status="failed", message="策略反馈与实际状态不一致：没有本次创建的日程或提醒。")
        if final.status == "failed":
            return Final(status="failed", message="本次未创建日程或提醒，任务未完成。" + self._last_error())
        return final

    def artifact(self):
        return {"schema_version": 1, "backend": self.policy.label, "scenario": self.env.scenario.name,
                "context": self.env.clock_now(None), "messages": self.messages, "trace": self.env.trace[self.start_trace_index:],
                "finals": self.finals, "raw_finals": self.raw_finals, "initial_state": self.initial_state,
                "state": self.env.snapshot(), "elapsed_seconds": self.elapsed_seconds,
                "metrics": {"policy_seconds": self.policy_seconds, "tool_seconds": self.tool_seconds,
                            "step_limit_reached": self.step_limit_reached}}


class Session:
    """Explicit new user intents (including update/undo) share state, never policy memory."""
    def __init__(self, env: Environment, policy_factory):
        self.env, self.policy_factory = env, policy_factory
        self.turns: list[Runtime] = []

    def submit(self, text: str, goal: GoalSpec) -> Runtime:
        runtime = Runtime(self.env, self.policy_factory(), goal=goal)
        runtime.run(text)
        self.turns.append(runtime)
        return runtime
