"""Reference controller reads structured USER intent, never hidden goals/weather/preferences."""
from __future__ import annotations

import copy

from .environment import TaskEnvironment


class ReferencePolicy:
    def __init__(self, intent):
        self.intent = copy.deepcopy(intent)
        self.followup = None

    def _call(self, env, name, args, allow_error=False):
        # The reference controller has no state/goal shortcut: every operation uses step.
        for attempt in range(2):
            response = env.step({"type": "tool_call", "name": name, "arguments": copy.deepcopy(args)})
            observation = response["observation"]
            reply = observation.get("user_reply")
            if reply and reply.get("slots", {}).get("intent"):
                self.followup = copy.deepcopy(reply["slots"])
            result = observation["tool_result"]
            if result["status"] == "error" and result["error_code"] == "TEMPORARY_UNAVAILABLE" and attempt == 0:
                continue
            if result["status"] == "error" and not allow_error:
                raise ValueError("参考调用失败: " + name + " " + result["error_code"])
            return result
        raise ValueError("参考策略重试异常")

    def _target(self, env):
        events = self._call(env, "list_events", self.intent["window"])["data"]["events"]
        selector = self.intent["target_selector"]
        matches = [e for e in events if all(e.get(k) == v for k, v in selector.items())]
        if len(matches) != 1:
            raise ValueError("可见请求未唯一确定目标事件")
        return matches[0]

    def _ask(self, env, slot):
        response = env.step({"type": "ask_user", "question": "请补充活动的" + ("开始时间。" if slot == "start_time" else "持续分钟数。"), "requested_slots": [slot]})
        reply = response["observation"]["user_reply"]
        if slot not in reply.get("slots", {}):
            raise ValueError("用户未提供必要参数")
        self.intent[slot] = reply["slots"][slot]

    def solve(self, env):
        intent, kind = self.intent, self.intent["kind"]
        event, reminder, query = None, None, None
        if kind in {"weather_query", "retry_weather"}:
            query = self._call(env, "get_weather", {"city": intent["city"], "date": intent["date"]})["data"]
        elif kind == "events_query":
            query = self._call(env, "list_events", intent["window"])["data"]
        elif kind in {"existing_reminder", "update", "delete"}:
            event = self._target(env)
            if kind == "existing_reminder":
                reminder = self._call(env, "set_reminder", {"event_id": event["event_id"], "minutes_before": intent["minutes_before"]})["data"]
            elif kind == "update":
                event = self._call(env, "update_event", {"event_id": event["event_id"], "start_time": intent["start_time"]})["data"]
            else:
                self._call(env, "delete_event", {"event_id": event["event_id"]})
        else:
            if kind in {"weather_activity", "weather_reschedule_remind"}:
                forecast = self._call(env, "get_weather", {"city": intent["city"], "date": intent["date"]})["data"]
                intent["activity_id"] = intent["weather_options"]["rain" if forecast["condition"] == "rain" else "otherwise"]
            elif kind == "preferences_activity":
                preferences = self._call(env, "get_preferences", {})["data"]["preferences"]
                candidates = self._call(env, "search_activity", {"indoor": preferences["indoor"], "max_duration_minutes": preferences["max_duration_minutes"]})["data"]["activities"]
                selected = min(candidates, key=lambda a: (a["duration_minutes"], a["activity_id"]))
                intent["activity_id"], intent["duration_minutes"] = selected["activity_id"], selected["duration_minutes"]
            elif kind == "clarify_time":
                self._ask(env, "start_time")
            elif kind == "clarify_duration":
                self._ask(env, "duration_minutes")
            times = [intent["start_time"]] + intent.get("alternatives", [])
            for start in times:
                created = self._call(env, "create_event", {"activity_id": intent["activity_id"], "start_time": start, "duration_minutes": intent["duration_minutes"]}, allow_error=True)
                if created["status"] == "success":
                    event = created["data"]
                    break
                if created["error_code"] != "TIME_CONFLICT":
                    raise ValueError("创建出现非冲突错误")
            if event is None:
                raise ValueError("所有用户授权时间都不可用")
            if self.followup:
                if self.followup["intent"] == "cancel":
                    self._call(env, "delete_event", {"event_id": event["event_id"]})
                elif self.followup["intent"] == "reschedule":
                    event = self._call(env, "update_event", {"event_id": event["event_id"], "start_time": self.followup["start_time"]})["data"]
                    if "minutes_before" in self.followup:
                        reminder = self._call(env, "set_reminder", {"event_id": event["event_id"], "minutes_before": self.followup["minutes_before"]})["data"]
            elif kind == "create_reminder":
                reminder = self._call(env, "set_reminder", {"event_id": event["event_id"], "minutes_before": intent["minutes_before"]})["data"]
        if query is not None:
            result, message = query, "已查询，结果见结构化事实。"
        else:
            result = {"event_id": event["event_id"]}
            if reminder:
                result["reminder_id"] = reminder["reminder_id"]
            message = "已按最后有效指令完成本地模拟操作；没有真实通知推送。"
        env.step({"type": "final", "status": "success", "result": result, "message": message})


def run_reference(task):
    env = TaskEnvironment(task)
    ReferencePolicy(task["reference_intent"]).solve(env)
    return env


def replay(task, trajectory):
    env = TaskEnvironment(task)
    if trajectory["case_id"] != task["case_id"]:
        raise ValueError("轨迹引用了其他任务")
    for row in trajectory["actions"]:
        actual = env.step(row["action"])["observation"]
        if actual != row["observation"]:
            raise ValueError("重放工具结果或用户回复不一致")
    if env._state != trajectory["final_state"] or env.evaluate() != trajectory["validation"]:
        raise ValueError("重放终态或判定不一致")
    if env.observation()["messages"] != trajectory["messages"]:
        raise ValueError("重放消息不一致")
    return env
