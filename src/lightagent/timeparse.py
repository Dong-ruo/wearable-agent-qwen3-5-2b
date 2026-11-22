"""Deliberately bounded Chinese time grammar; ambiguity is an error, never a guess."""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


class TimeResolutionError(ValueError):
    pass


def number(value: str) -> int:
    if value.isdigit():
        return int(value)
    digits = {c: n for n, c in enumerate("零一二三四五六七八九")}
    digits.update({"两": 2, "〇": 0})
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十")
        return (digits[left] if left else 1) * 10 + (digits[right] if right else 0)
    if value in digits:
        return digits[value]
    raise TimeResolutionError("无法识别数字，请使用阿拉伯数字。")


NUM = r"[0-9零〇一二两三四五六七八九十]{1,3}"
TIME_PATTERN = re.compile(
    rf"(?P<day>\d{{4}}-\d{{2}}-\d{{2}}|大后天|后天|明天|今天)\s*"
    rf"(?P<period>凌晨|早上|上午|中午|下午|晚上)?\s*"
    rf"(?P<hour>{NUM})(?:[:：](?P<colon>\d{{2}})|点(?P<minute>半|{NUM}分?)?)"
)


def resolve(expression: str, now: datetime, tz_name: str) -> datetime:
    tz = ZoneInfo(tz_name)
    local_now = now.astimezone(tz)
    matches = list(TIME_PATTERN.finditer(expression))
    if len(matches) != 1:
        raise TimeResolutionError("请提供一个明确时间，例如“明天下午六点”或“2026-11-17 18:00”。")
    m = matches[0]
    day = m["day"]
    if "-" in day:
        target_date = datetime.strptime(day, "%Y-%m-%d").date()
    else:
        target_date = local_now.date() + timedelta(days={"今天": 0, "明天": 1, "后天": 2, "大后天": 3}[day])
    hour = number(m["hour"])
    minute_text = m["colon"] or m["minute"]
    minute = 30 if minute_text == "半" else number(minute_text.rstrip("分")) if minute_text else 0
    period = m["period"]
    if period and not 1 <= hour <= 12 and not (hour == 0 and period == "凌晨"):
        raise TimeResolutionError("带上午/下午的时间请使用 1～12 点。")
    if (period == "晚上" and hour < 6) or (period in {"早上", "上午", "晚上"} and hour == 12):
        raise TimeResolutionError("该时段容易产生跨日或上午/下午歧义，请使用明确日期和 24 小时制。")
    if period in {"下午", "晚上"}:
        hour = hour % 12 + 12
    elif period == "中午":
        if hour not in {11, 12, 1}:
            raise TimeResolutionError("中午时间不明确，请使用 24 小时制。")
        hour = 13 if hour == 1 else hour
    elif period in {"凌晨", "早上", "上午"}:
        hour = hour % 12
    elif not m["colon"] and 1 <= hour <= 12:
        raise TimeResolutionError("请说明上午还是下午，或使用 24 小时制，例如 18:00。")
    try:
        naive = datetime.combine(target_date, time(hour, minute))
    except ValueError as exc:
        raise TimeResolutionError("小时或分钟超出范围。") from exc
    first = naive.replace(tzinfo=tz, fold=0)
    second = naive.replace(tzinfo=tz, fold=1)
    valid = [d for d in (first, second) if d.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) == naive]
    if not valid:
        raise TimeResolutionError("该当地时间因夏令时跳变不存在，请更换时间。")
    if len({d.utcoffset() for d in valid}) > 1:
        raise TimeResolutionError("该当地时间因夏令时重复，请明确其他时间。")
    if first <= local_now:
        raise TimeResolutionError("该时间已经过去，请提供将来的时间。")
    return first

