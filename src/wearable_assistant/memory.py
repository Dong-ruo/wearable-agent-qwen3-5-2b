"""Explicit, scoped memory for the local scheduling assistant."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from .assistance import clean_request


def constraint_change(text, now):
    text = clean_request(text).rstrip("。！!").strip()
    enable = re.fullmatch(r"(这周|本周|以后)只(?:安排|做)室内活动", text)
    disable = re.fullmatch(r"(?:取消(这周|本周|以后)的?室内活动限制|(这周|本周|以后)(?:也)?可以安排户外活动|取消室内活动限制)", text)
    if not enable and not disable:
        return None
    if not enable and not any(disable.groups()):
        return {"scope": "all", "enabled": False}
    scope = enable[1] if enable else next(group for group in disable.groups() if group)
    if scope == "以后":
        return {"scope": "session", "enabled": bool(enable)}
    day = datetime.fromisoformat(now).date()
    first = day - timedelta(days=day.weekday())
    return {"scope": "week", "enabled": bool(enable), "from_date": first.isoformat(),
            "through_date": (first + timedelta(days=6)).isoformat()}


def indoor_applies(constraints, start_time):
    day = datetime.fromisoformat(start_time).date().isoformat()
    return any(rule["enabled"] and (rule["scope"] == "session"
               or (rule["scope"] == "week" and rule["from_date"] <= day <= rule["through_date"]))
               for rule in constraints.values())


def candidate_events(objects, selector, now):
    from .followups import CLOCK, explicit_event_id, resolve_start

    if not selector or selector in {"活动", "该活动", "这个活动", "这项活动", "这条活动"}:
        return list(objects.values())
    event_id = explicit_event_id(selector)
    if event_id:
        return [objects[event_id]] if event_id in objects else []
    time_and_title = re.fullmatch(rf"(?P<clock>{CLOCK})(?:开始的|的)?\s*(?P<title>[^，。；、,]+)?", selector)
    if not time_and_title:
        title = selector.strip('“”"')
        return [event for event in objects.values() if event.get("title") == title]
    title = (time_and_title["title"] or "").strip('“”" ')
    matches = []
    for event in objects.values():
        if title and title != "活动" and event.get("title") != title:
            continue
        try:
            if datetime.fromisoformat(event["start_time"]) == datetime.fromisoformat(
                    resolve_start(time_and_title["clock"], event["start_time"], now)):
                matches.append(event)
        except ValueError:
            continue
    return matches


def mentioned_event_ids(text, objects, now):
    ids = set(re.findall(r"\bevent_[0-9]+\b", text)) & objects.keys()
    if ids:
        return ids
    titles = {event_id for event_id, event in objects.items() if event.get("title") and event["title"] in text}
    from .followups import CLOCK
    times = set()
    for match in re.finditer(CLOCK, text):
        times.update(event["event_id"] for event in candidate_events(objects, match[0], now))
    return (titles & times) or titles or times
