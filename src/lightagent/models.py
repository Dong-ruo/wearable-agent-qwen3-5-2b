from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Empty(StrictModel):
    pass


class ResolveArgs(StrictModel):
    expression: str = Field(min_length=1, max_length=200)


class WindowArgs(StrictModel):
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("时间必须包含 UTC 偏移量")
        return value

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("end 必须晚于 start")
        return self


class SlotArgs(WindowArgs):
    duration_minutes: int = Field(strict=True, ge=5, le=240)


class WeatherArgs(StrictModel):
    city: str = Field(min_length=1, max_length=40)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class EventArgs(StrictModel):
    title: str = Field(min_length=1, max_length=80)
    activity: Literal["outdoor_run", "indoor_stretch", "meeting", "other"]
    city: str = Field(min_length=1, max_length=40)
    start: datetime
    duration_minutes: int = Field(strict=True, ge=5, le=240)
    idempotency_key: str = Field(min_length=1, max_length=100)

    @field_validator("start")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return WindowArgs.aware(value)


class EventId(StrictModel):
    event_id: str = Field(min_length=1, max_length=100)


class ReminderArgs(EventId):
    minutes_before: int = Field(strict=True, ge=0, le=1440)
    idempotency_key: str = Field(min_length=1, max_length=100)


class ReminderId(StrictModel):
    reminder_id: str = Field(min_length=1, max_length=100)


class UpdateArgs(EventId):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    activity: Literal["outdoor_run", "indoor_stretch", "meeting", "other"] | None = None
    city: str | None = Field(default=None, min_length=1, max_length=40)
    start: datetime | None = None
    duration_minutes: int | None = Field(default=None, strict=True, ge=5, le=240)
    expected_revision: int | None = Field(default=None, strict=True, ge=0)
    idempotency_key: str = Field(min_length=1, max_length=100)

    @field_validator("start")
    @classmethod
    def aware(cls, value):
        return WindowArgs.aware(value) if value is not None else value

    @model_validator(mode="after")
    def has_change(self):
        if not any(getattr(self, key) is not None for key in ("title", "activity", "city", "start", "duration_minutes")):
            raise ValueError("至少提供一个待修改字段")
        return self


class DeleteArgs(EventId):
    idempotency_key: str = Field(min_length=1, max_length=100)
    expected_revision: int | None = Field(default=None, strict=True, ge=0)


class ActivityArgs(StrictModel):
    indoor: bool | None = Field(default=None, strict=True)
    max_duration_minutes: int = Field(default=240, strict=True, ge=5, le=240)


class AskArgs(StrictModel):
    question: str = Field(min_length=1, max_length=500)
    slots: list[str] = Field(min_length=1, max_length=8)


class Call(StrictModel):
    name: str
    arguments: dict[str, Any]


class Final(StrictModel):
    status: Literal["success", "partial", "failed", "clarification"]
    message: str = Field(min_length=1, max_length=2000)
    event_id: str | None = None
    reminder_id: str | None = None
    requested_slots: list[str] = Field(default_factory=list)


class Scenario(StrictModel):
    name: str
    now: datetime
    timezone: str = "Asia/Shanghai"
    events: list[dict[str, Any]] = Field(default_factory=list)
    weather: dict[str, str] = Field(default_factory=dict)
    faults: dict[str, Any] = Field(default_factory=dict)
    reminders: list[dict[str, Any]] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    activities: list[dict[str, Any]] = Field(default_factory=lambda: [
        {"id": "outdoor_run", "title": "户外跑步", "indoor": False, "duration_minutes": 30},
        {"id": "indoor_stretch", "title": "室内拉伸", "indoor": True, "duration_minutes": 30},
    ])

    @field_validator("now")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return WindowArgs.aware(value)

