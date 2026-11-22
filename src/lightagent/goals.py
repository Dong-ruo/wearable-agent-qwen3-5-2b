"""Private, declarative goal predicates; nothing here generates task instances."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .environment import Environment, dt
from .models import Scenario, StrictModel, WindowArgs


class EventTarget(StrictModel):
    event_id: str | None = None
    title: str | None = None
    activity: str | None = None
    city: str | None = None
    start: datetime | None = None
    duration_minutes: int | None = Field(default=None, gt=0)
    count: int = Field(default=1, ge=1)

    @field_validator("start")
    @classmethod
    def aware(cls, value):
        return WindowArgs.aware(value) if value is not None else value

    @model_validator(mode="after")
    def specific(self):
        if not any(getattr(self, k) is not None for k in ("event_id", "title", "activity", "city", "start", "duration_minutes")):
            raise ValueError("事件目标至少应指定一个可校验字段")
        return self

    def matches(self, event):
        fields = {"id": self.event_id, "title": self.title, "activity": self.activity, "city": self.city}
        return (all(value is None or event.get(key) == value for key, value in fields.items()) and
                (self.start is None or dt(event["start"]) == self.start) and
                (self.duration_minutes is None or dt(event["end"]) - dt(event["start"]) == timedelta(minutes=self.duration_minutes)))


class ReminderTarget(StrictModel):
    event_id: str | None = None
    event_target_index: int | None = Field(default=None, ge=0)
    minutes_before: int = Field(ge=0, le=1440)
    count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def link(self):
        if (self.event_id is None) == (self.event_target_index is None):
            raise ValueError("提醒必须指定 event_id 或 event_target_index，二选一")
        return self


class WeatherTarget(StrictModel):
    city: str
    date: str
    condition: str | None = None


class GoalSpec(StrictModel):
    events: list[EventTarget] = Field(default_factory=list)
    reminders: list[ReminderTarget] = Field(default_factory=list)
    deleted_event_ids: list[str] = Field(default_factory=list)
    weather: list[WeatherTarget] = Field(default_factory=list)
    new_event_count: int = Field(default=0, ge=0)
    new_reminder_count: int = Field(default=0, ge=0)
    preserve_unmentioned: bool = True
    require_verification: bool = True
    require_no_overlap: bool = True

    @model_validator(mode="after")
    def coherent(self):
        if not (self.events or self.reminders or self.deleted_event_ids or self.weather):
            raise ValueError("目标不能是空的")
        for reminder in self.reminders:
            if reminder.event_target_index is not None and reminder.event_target_index >= len(self.events):
                raise ValueError("提醒关联的目标索引不存在")
        return self


class EpisodeSpec(StrictModel):
    request: str = Field(min_length=1)
    scenario: Scenario
    goal: GoalSpec
    missing_slots: set[str] = Field(default_factory=set)
    user_answers: dict[str, str] = Field(default_factory=dict)
    family_id: str | None = None
    split: Literal["unspecified", "train", "validation", "test"] = "unspecified"
    test_category: Literal["in_distribution", "compositional", "robustness"] | None = None

    @model_validator(mode="after")
    def user_state(self):
        if not self.missing_slots <= self.user_answers.keys():
            raise ValueError("缺失参数没有对应的用户模拟回答")
        return self


def evaluate_goal(goal: GoalSpec, env: Environment, trace_start: int = 0, initial_state: dict | None = None) -> dict[str, bool]:
    initial = initial_state or {"events": env.scenario.events, "reminders": env.scenario.reminders}
    trace = env.trace[trace_start:]
    checks = {}
    initial_ids = {e["id"] for e in initial["events"]}
    initial_rids = {r["id"] for r in initial["reminders"]}
    checks["new_event_count"] = sum(e["id"] not in initial_ids and e.get("origin") == "agent" for e in env.events) == goal.new_event_count
    checks["new_reminder_count"] = sum(r["id"] not in initial_rids for r in env.reminders) == goal.new_reminder_count
    matched = [[e for e in env.events if target.matches(e)] for target in goal.events]

    def verified(name, item, writes):
        mutations = [r["index"] for r in trace if r["name"] in writes and r["result"]["ok"] and r["result"]["data"].get("id") == item["id"]]
        last_write = max(mutations, default=-1)
        return any(r["name"] == name and r["result"]["ok"] and r["index"] > last_write and r["result"]["data"] == item for r in trace)

    for i, (target, matches) in enumerate(zip(goal.events, matched)):
        checks[f"event_{i}_match"] = len(matches) == target.count
        if goal.require_verification:
            checks[f"event_{i}_verified"] = bool(matches) and all(verified("calendar_get", e, {"calendar_create", "calendar_update"}) for e in matches)
        if goal.require_no_overlap:
            checks[f"event_{i}_no_overlap"] = all(not any(other["id"] != e["id"] and dt(other["start"]) < dt(e["end"]) and dt(other["end"]) > dt(e["start"]) for other in env.events) for e in matches)
    for i, target in enumerate(goal.reminders):
        ids = {target.event_id} if target.event_id is not None else {e["id"] for e in matched[target.event_target_index]}
        reminders = [r for r in env.reminders if r["event_id"] in ids and r["minutes_before"] == target.minutes_before]
        checks[f"reminder_{i}_match"] = len(reminders) == target.count
        checks[f"reminder_{i}_time"] = bool(reminders) and all(any(e["id"] == r["event_id"] and dt(r["remind_at"]) == dt(e["start"]) - timedelta(minutes=r["minutes_before"]) for e in env.events) for r in reminders)
        if goal.require_verification:
            # Updating a linked event also moves its reminder; require the read after that move.
            checks[f"reminder_{i}_verified"] = bool(reminders) and all(verified("reminder_get", r, {"reminder_create"}) and
                any(row["name"] == "reminder_get" and row["result"]["ok"] and row["result"]["data"] == r and
                    row["index"] > max((x["index"] for x in trace if x["name"] == "calendar_update" and x["result"]["ok"] and x["result"]["data"]["id"] == r["event_id"]), default=-1) for row in trace) for r in reminders)
    for event_id in goal.deleted_event_ids:
        checks[f"deleted_{event_id}"] = not any(e["id"] == event_id for e in env.events) and not any(r["event_id"] == event_id for r in env.reminders)
        if goal.require_verification:
            writes = [r["index"] for r in trace if r["name"] == "calendar_delete" and r["result"]["ok"] and r["arguments"]["event_id"] == event_id]
            last = max(writes, default=-1)
            original = next((e for e in initial["events"] if e["id"] == event_id), None)
            checks[f"deleted_{event_id}_verified"] = bool(writes) and any(r["index"] > last and (
                (r["name"] == "calendar_get" and r["arguments"]["event_id"] == event_id and not r["result"]["ok"] and r["result"]["error"]["code"] == "not_found") or
                (r["name"] == "calendar_list" and r["result"]["ok"] and original is not None and dt(r["arguments"]["start"]) <= dt(original["start"]) and dt(r["arguments"]["end"]) >= dt(original["end"]) and not any(e["id"] == event_id for e in r["result"]["data"]["events"]))
            ) for r in trace)
    for i, target in enumerate(goal.weather):
        checks[f"weather_{i}"] = any(r["name"] == "weather_get" and r["result"]["ok"] and r["result"]["data"]["city"] == target.city and r["result"]["data"]["date"] == target.date and (target.condition is None or r["result"]["data"]["condition"] == target.condition) for r in trace)
    if goal.preserve_unmentioned:
        mutable = set(goal.deleted_event_ids) | {e.event_id for e in goal.events if e.event_id}
        checks["protected_events"] = all(e in env.events for e in initial["events"] if e["id"] not in mutable)
        mutable_reminder_events = mutable | {r.event_id for r in goal.reminders if r.event_id}
        checks["protected_reminders"] = all(r in env.reminders for r in initial["reminders"] if r["event_id"] not in mutable_reminder_events)
    return checks
