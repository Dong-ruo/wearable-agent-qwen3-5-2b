from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .models import (ActivityArgs, AskArgs, DeleteArgs, Empty, EventArgs, EventId, Final, ReminderArgs, ReminderId, ResolveArgs, UpdateArgs,
                     Scenario, SlotArgs, WeatherArgs, WindowArgs)
from .timeparse import resolve


SPECS = {
    "clock_now": (Empty, "获取环境当前日期时间和时区。"),
    "time_resolve": (ResolveArgs, "依据当前日期和时区解析中文时间；歧义或过去时间返回错误。"),
    "calendar_list": (WindowArgs, "查询半开区间 [start,end) 内所有冲突日程。"),
    "weather_get": (WeatherArgs, "查询模拟天气：sunny/cloudy/rain；缺少记录会返回错误。"),
    "calendar_find_slot": (SlotArgs, "在指定窗口内寻找最早可用时段，只查询，不创建。"),
    "calendar_create": (EventArgs, "创建活动；冲突时失败。必须使用稳定幂等键。"),
    "calendar_get": (EventId, "读取创建后的日程以核验实际状态。"),
    "reminder_create": (ReminderArgs, "为已有日程创建提醒；可独立失败，必须核验。"),
    "reminder_get": (ReminderId, "读取提醒以核验时间和关联日程。"),
    "calendar_update": (UpdateArgs, "修改指定日程；冲突或提醒落在过去时整次拒绝。关联提醒随时间移动。"),
    "calendar_delete": (DeleteArgs, "删除指定日程及关联提醒；只能在用户明确要求后调用。"),
    "activity_search": (ActivityArgs, "按室内/室外和最长时长筛选活动目录，仅返回候选，不替用户选择。"),
    "preferences_get": (Empty, "读取用户已配置的偏好；缺少的偏好不会被推测。"),
    "user_ask": (AskArgs, "向已配置的确定性用户询问缺失信息，只返回被明确询问的必要字段；未配置时不可用。"),
}


def tool_specs() -> list[dict]:
    return [{"type": "function", "function": {"name": name, "description": desc,
             "parameters": model.model_json_schema()}} for name, (model, desc) in SPECS.items()]


class ToolError(ValueError):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code, self.details = code, details


def dt(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("环境状态必须使用有时区的时间")
    return result


class Environment:
    """Each rollout owns a deep copy; observations never contain hidden fault switches."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario.model_copy(deep=True)
        self.now = scenario.now.astimezone(ZoneInfo(scenario.timezone))
        self.events = copy.deepcopy(scenario.events)
        self.reminders: list[dict] = copy.deepcopy(scenario.reminders)
        self.faults = copy.deepcopy(scenario.faults)
        self.trace: list[dict] = []
        self.idempotency: dict[tuple[str, str], tuple[dict, dict]] = {}
        self._used_ids = {e["id"] for e in self.events} | {r["id"] for r in self.reminders}
        self._user = None
        ids = [e["id"] for e in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("初始日程 ID 重复")
        for e in self.events:
            if dt(e["end"]) <= dt(e["start"]):
                raise ValueError("初始日程结束时间必须晚于开始时间")

    def execute(self, name: str, arguments: dict[str, Any]) -> dict:
        try:
            if name not in SPECS:
                raise ToolError("unknown_tool", "工具不存在")
            args = SPECS[name][0].model_validate(arguments)
            unavailable = self.faults.get("unavailable", [])
            if name in unavailable:
                raise ToolError("unavailable", "模拟服务暂不可用")
            data = getattr(self, name)(args)
            result = {"ok": True, "data": data}
        except ValidationError as exc:
            result = {"ok": False, "error": {"code": "invalid_arguments", "message": str(exc)}}
        except ToolError as exc:
            result = {"ok": False, "error": {"code": exc.code, "message": str(exc), **exc.details}}
        except (ValueError, KeyError) as exc:
            result = {"ok": False, "error": {"code": "invalid_value", "message": str(exc)}}
        self.trace.append(copy.deepcopy({"index": len(self.trace), "name": name,
                                       "arguments": arguments, "result": result}))
        return result

    def clock_now(self, _):
        return {"now": self.now.isoformat(), "timezone": self.scenario.timezone, "simulated": True}

    def time_resolve(self, args):
        return {"start": resolve(args.expression, self.now, self.scenario.timezone).isoformat(),
                "timezone": self.scenario.timezone}

    def conflicts(self, start: datetime, end: datetime) -> list[dict]:
        return [copy.deepcopy(e) for e in self.events if dt(e["start"]) < end and dt(e["end"]) > start]

    def calendar_list(self, args):
        return {"events": self.conflicts(args.start, args.end)}

    def weather_get(self, args):
        key = f"{args.city}|{args.date}"
        condition = self.scenario.weather.get(key)
        if condition not in {"sunny", "cloudy", "rain"}:
            raise ToolError("weather_missing", "该城市/日期没有模拟天气记录，请补充环境配置或稍后重试")
        return {"city": args.city, "date": args.date, "condition": condition, "simulated": True}

    def calendar_find_slot(self, args):
        candidate = max(args.start, self.now)
        duration = timedelta(minutes=args.duration_minutes)
        while candidate + duration <= args.end:
            conflicts = self.conflicts(candidate, candidate + duration)
            if not conflicts:
                return {"start": candidate.isoformat(), "end": (candidate + duration).isoformat()}
            candidate = max(dt(e["end"]) for e in conflicts)
        raise ToolError("no_slot", "查询窗口内没有足够长的空闲时段")

    def _cached(self, kind, args):
        key = (kind, args.idempotency_key)
        payload = args.model_dump(mode="json")
        if key in self.idempotency:
            previous, result = self.idempotency[key]
            if previous != payload:
                raise ToolError("idempotency_mismatch", "同一幂等键不能用于不同参数")
            return copy.deepcopy(result)
        return None

    def _remember(self, kind, args, result):
        self.idempotency[(kind, args.idempotency_key)] = (args.model_dump(mode="json"), copy.deepcopy(result))
        return result

    def _fail(self, key):
        if self.faults.get(key, 0) > 0:
            self.faults[key] -= 1
            raise ToolError("write_failed", "模拟写入失败，未写入任何状态")

    def _id(self, prefix, items):
        used = self._used_ids | {x["id"] for x in items}
        n = 1
        while f"{prefix}{n}" in used:
            n += 1
        value = f"{prefix}{n}"
        self._used_ids.add(value)
        return value

    def calendar_create(self, args):
        cached = self._cached("event", args)
        if cached is not None:
            return cached
        if args.start <= self.now:
            raise ToolError("past_time", "不能创建过去的活动")
        end = args.start + timedelta(minutes=args.duration_minutes)
        race = self.faults.pop("race_before_create", None)
        if race:
            self.events.append({"id": self._id("race-event-", self.events), "title": "临时会议", "start": args.start.isoformat(),
                                "end": end.isoformat(), "origin": "external"})
        conflicts = self.conflicts(args.start, end)
        if conflicts:
            raise ToolError("conflict", "该时间段已有安排", events=conflicts)
        self._fail("event_create_failures")
        event = {"id": self._id("event-", self.events), "title": args.title, "activity": args.activity,
                 "city": args.city, "start": args.start.isoformat(), "end": end.isoformat(), "origin": "agent"}
        self.events.append(event)
        return self._remember("event", args, copy.deepcopy(event))

    def calendar_get(self, args):
        event = next((e for e in self.events if e["id"] == args.event_id), None)
        if event is None:
            raise ToolError("not_found", "找不到日程")
        return copy.deepcopy(event)

    def reminder_create(self, args):
        cached = self._cached("reminder", args)
        if cached is not None:
            return cached
        event = self.calendar_get(EventId(event_id=args.event_id))
        remind_at = dt(event["start"]) - timedelta(minutes=args.minutes_before)
        if remind_at <= self.now:
            raise ToolError("past_reminder", "提醒时间已经过去，请确认其他提前时间")
        self._fail("reminder_create_failures")
        existing = next((r for r in self.reminders if r["event_id"] == args.event_id), None)
        if existing:
            raise ToolError("duplicate_reminder", "该日程已有提醒")
        reminder = {"id": self._id("reminder-", self.reminders), "event_id": args.event_id,
                    "minutes_before": args.minutes_before, "remind_at": remind_at.isoformat()}
        self.reminders.append(reminder)
        return self._remember("reminder", args, copy.deepcopy(reminder))

    def reminder_get(self, args):
        reminder = next((r for r in self.reminders if r["id"] == args.reminder_id), None)
        if reminder is None:
            raise ToolError("not_found", "找不到提醒")
        return copy.deepcopy(reminder)

    def calendar_update(self, args):
        cached = self._cached("update", args)
        if cached is not None:
            return cached
        event = self.calendar_get(EventId(event_id=args.event_id))
        if args.expected_revision is not None and event.get("revision", 0) != args.expected_revision:
            raise ToolError("stale_revision", "日程已被其他操作修改，请重新读取")
        start = args.start or dt(event["start"])
        duration = timedelta(minutes=args.duration_minutes) if args.duration_minutes is not None else dt(event["end"]) - dt(event["start"])
        if start <= self.now:
            raise ToolError("past_time", "不能移动到过去的时间")
        conflicts = [e for e in self.conflicts(start, start + duration) if e["id"] != args.event_id]
        if conflicts:
            raise ToolError("conflict", "修改后的时间段已有安排", events=conflicts)
        updated_reminders = copy.deepcopy(self.reminders)
        for reminder in updated_reminders:
            if reminder["event_id"] == args.event_id:
                at = start - timedelta(minutes=reminder["minutes_before"])
                if at <= self.now:
                    raise ToolError("past_reminder", "移动后提醒时间已过，修改未生效")
                reminder["remind_at"] = at.isoformat()
        for key in ("title", "activity", "city"):
            if getattr(args, key) is not None:
                event[key] = getattr(args, key)
        event.update(start=start.isoformat(), end=(start + duration).isoformat(), revision=event.get("revision", 0) + 1)
        self._fail("event_update_failures")
        self.events = [event if e["id"] == args.event_id else e for e in self.events]
        self.reminders = updated_reminders
        return self._remember("update", args, copy.deepcopy(event))

    def calendar_delete(self, args):
        cached = self._cached("delete", args)
        if cached is not None:
            return cached
        event = self.calendar_get(EventId(event_id=args.event_id))
        if args.expected_revision is not None and args.expected_revision != event.get("revision", 0):
            raise ToolError("stale_revision", "日程已变化，请重新读取")
        removed = [r["id"] for r in self.reminders if r["event_id"] == args.event_id]
        self._fail("event_delete_failures")
        self.events = [e for e in self.events if e["id"] != args.event_id]
        self.reminders = [r for r in self.reminders if r["event_id"] != args.event_id]
        return self._remember("delete", args, {"id": args.event_id, "deleted": True, "deleted_reminder_ids": removed})

    def activity_search(self, args):
        return {"activities": [copy.deepcopy(a) for a in self.scenario.activities
                if (args.indoor is None or a["indoor"] == args.indoor) and a["duration_minutes"] <= args.max_duration_minutes]}

    def preferences_get(self, _):
        return {"preferences": copy.deepcopy(self.scenario.preferences)}

    def user_ask(self, args):
        if self._user is None:
            raise ToolError("unavailable", "未配置用户模拟器，请通过 clarification 向真实用户询问")
        response = self._user.respond(Final(status="clarification", message=args.question, requested_slots=args.slots))
        if response is None:
            raise ToolError("unnecessary_question", "未询问尚缺失的必要信息，不提供额外参数")
        return {"user_response": response}

    def snapshot(self):
        return copy.deepcopy({"events": self.events, "reminders": self.reminders})
