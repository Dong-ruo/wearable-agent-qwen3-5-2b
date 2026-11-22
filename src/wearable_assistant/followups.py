"""Recognize explicit, single-operation local event follow-ups."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from .assistance import clean_request, resolve_day

EVENT_ID = r"event_[0-9]+"
CREATED_REFERENCES = ("刚创建的活动", "刚才创建的活动")
REFERENCE = rf"(?:event_id\s*=\s*)?{EVENT_ID}|[^，。；、,!?！？\n]{{1,70}}?"
CLOCK = r"(?:(?:今天|明天|后天)?\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00)"


def parse_followup(text):
    text = clean_request(text).rstrip("。？?！!").strip()
    match = re.fullmatch(rf"(?:请|帮我|请帮我)?给(?P<target>{REFERENCE})设置(?:提前\s*(?P<minutes>[0-9]{{1,4}})\s*分钟的?)?提醒", text)
    if match:
        return {"operation": "set_reminder", "reference": match["target"],
                "minutes_before": int(match["minutes"]) if match["minutes"] else None}
    match = re.fullmatch(rf"(?:请|帮我|请帮我)?(?:只)?把(?P<target>{REFERENCE})(?:的开始时间)?(?:改到(?P<clock>{CLOCK})|改期)(?:[，,]\s*(?:时长和活动不变|时长不变))?", text)
    if match:
        return {"operation": "update_event", "reference": match["target"], "clock": match["clock"]}
    match = re.fullmatch(rf"(?:请|帮我|请帮我)?(?:核对|查看|查询)(?:(?P<target>{REFERENCE})的)?(?:关联)?提醒", text)
    if match:
        return {"operation": "read_reminder", "reference": match["target"]}
    match = re.fullmatch(rf"(?P<target>{REFERENCE})(?:以后|后续)只改开始时间", text)
    if match:
        return {"operation": "protect_time", "reference": match["target"]}
    match = re.fullmatch(rf"取消(?P<target>{REFERENCE})(?:的)?只改开始时间限制", text)
    if match:
        return {"operation": "release_time", "reference": match["target"]}
    return None


def explicit_event_id(text):
    match = re.fullmatch(rf"(?:event_id\s*=\s*)?({EVENT_ID})[。]?", text.strip())
    return match[1] if match else None


def resolve_start(clock, current_start, now):
    match = re.fullmatch(r"(今天|明天|后天)?(\d{1,2}):(\d{2})", clock)
    if not match:
        return datetime.fromisoformat(clock).isoformat()
    date = resolve_day(match[1], now) if match[1] else datetime.fromisoformat(current_start).date().isoformat()
    return datetime.fromisoformat(date + "T00:00:00+08:00").replace(hour=int(match[2]), minute=int(match[3])).isoformat()


def reminder_matches(event, reminder, expected):
    if reminder is None:
        return expected is None
    if expected is None:
        return False
    try:
        expected_time = datetime.fromisoformat(event["start_time"]) - timedelta(minutes=expected["minutes_before"])
        return (reminder["event_id"] == event["event_id"]
                and reminder["reminder_id"] == expected["reminder_id"]
                and reminder["minutes_before"] == expected["minutes_before"]
                and datetime.fromisoformat(reminder["trigger_time"]) == expected_time)
    except (KeyError, TypeError, ValueError):
        return False
