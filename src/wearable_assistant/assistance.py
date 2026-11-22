"""Bounded public-request matching; no environment state, private goals or ID aliases."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from wearable_tasks.contract import canonical


def clean_request(text):
    return re.sub(r"^以模拟当前时间 \S+ 为准，", "", text).strip()


def resolve_day(value, now):
    if value in {"今天", "明天", "后天"}:
        if not now: return None
        return (datetime.fromisoformat(now) + timedelta(days={"今天": 0, "明天": 1, "后天": 2}[value])).date().isoformat()
    return value


def simple_list_arguments(text, now):
    match = re.fullmatch(r"(?:请|帮我)?(?:查询|查看|查一下|列出)(今天|明天|后天|\d{4}-\d{2}-\d{2})的?日程[。？?]*", clean_request(text))
    if not match: return None
    day = resolve_day(match[1], now)
    if not day: return None
    try:
        start = datetime.fromisoformat(day + "T00:00:00+08:00")
    except ValueError:
        return None
    return {"start_time": start.isoformat(), "end_time": (start + timedelta(days=1)).isoformat()}


def simple_query_final(text, name, arguments, data, now):
    text = clean_request(text)
    if name == "get_preferences" and not arguments and re.fullmatch(
            r"(?:请|帮我)?(?:查看|查询|看看|查一下)(?:我的)?(?:运动)?偏好[。？?]*", text):
        if isinstance(data.get("preferences"), dict):
            return "运动偏好查询结果：" + canonical(data["preferences"])
    if name == "list_events":
        expected = simple_list_arguments(text, now)
        if arguments != expected or not isinstance(data.get("events"), list): return None
        day = expected["start_time"][:10]
        return f"{day}暂无日程。" if not data["events"] else f"{day}查询到{len(data['events'])}条日程：" + canonical(data["events"])
    return None


def simple_create_request(text, now):
    clock = r"(?:今天|明天|后天)\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00"
    match = re.fullmatch(rf"(?:请|帮我)?(?P<clock>{clock})(?:安排|创建)(?P<minutes>\d{{1,3}})分钟(?P<title>[^，。,；;！？?！\n]{{1,40}})[。]*", clean_request(text))
    if not match or re.search(r"并(?:且)?(?:设|加|提|提醒|改|删)|随后|然后|再(?:设|加|提|提醒)|(?:不|别|不要)(?:设|加|提醒)", match["title"]):
        return None
    from .followups import resolve_start

    try:
        start = resolve_start(match["clock"], now, now)
    except (TypeError, ValueError):
        return None
    return {"start_time": start, "duration_minutes": int(match["minutes"]), "title": match["title"]}


WRITE_PATTERNS = {
    "create_event": r"(?:立即)?(?:安排|创建|新增|添加)(?:一个|一项|一条|今天|明天|后天|活动|日程|\d|[^\s，。])",
    "update_event": r"(?:把|将).{1,100}(?:改到|改为|改成|调到|改期)|(?:修改|调整|变更).{1,100}(?:时间|时长|活动|日程)",
    "delete_event": r"(?:取消|删除|撤销).{0,50}(?:活动|日程|event_[0-9]+)",
    "set_reminder": r"(?:给|为).{1,80}(?:设定|设置|添加|加上).{0,30}提醒|(?:设置|加上|设定).{0,30}提醒|(?:提前\s*\d{1,4}\s*分钟)?提醒我",
}


def requested_writes(text):
    """Return evidence obligations for explicit write requests, including simple compound requests."""
    text = clean_request(text)
    clauses = re.split(r"[，。；,;\n]|并且|然后|随后|并(?=给|为|设|添|改|取|删|提前|提醒)|再(?=给|为|设|添|改|安|创|取|删|提前|提醒)", text)
    operations = set()
    for clause in clauses:
        clause = re.sub(r"^(?:请帮我|帮我|请|先|只把|只)\s*", lambda m: "把" if m[0] == "只把" else "", clause.strip())
        if re.match(r"(?:不|别|禁止|如何|怎么|能否解释|查看|查询|核对|检查)", clause):
            continue
        if re.fullmatch(r"取消(?:本周|这周|以后)?的?室内活动限制|取消.+只改开始时间限制", clause):
            continue
        if re.fullmatch(r"(?:本周|这周|以后)(?:只|也可以|可以)安排(?:室内|户外)活动", clause):
            continue
        clause = re.sub(r"^(?:(?:今天|明天|后天)(?:\s*\d{1,2}:\d{2})?|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00)\s*", "", clause)
        operations.update(name for name, pattern in WRITE_PATTERNS.items() if re.match(pattern, clause))
    return operations


def simple_create_matches(messages, data, now):
    users = [clean_request(m["content"]) for m in messages if m["role"] == "user"]
    if len(users) not in {1, 2} or not isinstance(data.get("title"), str): return False
    clock = r"(?:今天|明天|后天)\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00"
    head = r"(?:请|帮我)?(?P<clock>" + clock + r")(?:安排|创建)"
    match = re.fullmatch(head + r"(?P<minutes>\d+)分钟" + re.escape(data["title"]) + r"[。]*", users[0])
    if match and len(users) != 1: return False
    if not match and len(users) == 2:
        match = re.fullmatch(head + re.escape(data["title"]) + r"[，,]时长请先问我[。]*", users[0])
        reply = re.fullmatch(r"(?:持续|时长(?:为)?)?(\d+)分钟[。]*", users[1])
        if not match or not reply: return False
        minutes = int(reply[1])
    elif match:
        minutes = int(match["minutes"])
    else: return False
    value = match["clock"]
    relative = re.fullmatch(r"(今天|明天|后天)(\d{1,2}):(\d{2})", value)
    try:
        if relative:
            day = resolve_day(relative[1], now)
            if not day: return False
            value = datetime.fromisoformat(day + "T00:00:00+08:00").replace(hour=int(relative[2]), minute=int(relative[3])).isoformat()
        return value == data.get("start_time") and minutes == data.get("duration_minutes")
    except ValueError: return False
