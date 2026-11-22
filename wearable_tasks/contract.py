"""Frozen eight-tool contract; adapted from lightagent's interval/atomic-write semantics."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
READ_ONLY = frozenset({"list_events", "get_weather", "search_activity", "get_preferences"})
WRITES = frozenset({"create_event", "update_event", "delete_event", "set_reminder"})


def stamp(value):
    if not isinstance(value, str):
        raise ValueError("时间戳必须是字符串")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != timedelta(hours=8):
        raise ValueError("时间戳必须显式使用 +08:00")
    return result


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _string(description, fmt=None):
    value = {"type": "string", "minLength": 1, "maxLength": 200, "description": description}
    if fmt:
        value["format"] = fmt
    return value


def _integer(description, minimum=1, maximum=240):
    return {"type": "integer", "minimum": minimum, "maximum": maximum, "description": description}


def _tool(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}}}


_SCHEMAS = (
    _tool("list_events", "查询与半开区间 [start_time,end_time) 相交的日程，按开始时间与 ID 稳定排序。",
          {"start_time": _string("查询开始时间", "date-time"), "end_time": _string("查询结束时间", "date-time")}, ("start_time", "end_time")),
    _tool("create_event", "按明确参数创建日程；时间冲突时失败，不自动选择其他时间。",
          {"activity_id": _string("活动目录中的 ID"), "start_time": _string("开始时间", "date-time"), "duration_minutes": _integer("持续分钟数", 5)}, ("activity_id", "start_time", "duration_minutes")),
    _tool("update_event", "原子修改已有事件；冲突时排除自身。关联提醒随时间移动，失败不写入。",
          {"event_id": _string("已有事件 ID"), "activity_id": _string("新的活动 ID"), "start_time": _string("新的开始时间", "date-time"), "duration_minutes": _integer("新的持续分钟数", 5)}, ("event_id",)),
    _tool("delete_event", "删除指定日程及关联提醒；不存在的 ID 返回 NOT_FOUND。",
          {"event_id": _string("待删除的已有事件 ID")}, ("event_id",)),
    _tool("get_weather", "读取固定天气表，只返回事实；缺少数据时返回 NOT_FOUND。",
          {"city": _string("查询城市"), "date": _string("YYYY-MM-DD 日期", "date")}, ("city", "date")),
    _tool("search_activity", "按明确 ID、室内属性、目录时长筛选活动，不替用户作规划。",
          {"activity_id": _string("指定活动 ID"), "indoor": {"type": "boolean", "description": "是否室内"}, "max_duration_minutes": _integer("允许的最长目录时长", 5)}),
    _tool("set_reminder", "为事件设置唯一提醒；相同输入保持幂等。触发时间不早于模拟当前时间。",
          {"event_id": _string("已有事件 ID"), "minutes_before": _integer("提前分钟数，必须大于零", 1, 1440)}, ("event_id", "minutes_before")),
    _tool("get_preferences", "仅返回当前任务允许查询的已配置偏好。", {}),
)


def tool_schemas(order=None):
    tools = copy.deepcopy(list(_SCHEMAS))
    if order is not None:
        mapping = {t["function"]["name"]: t for t in tools}
        if len(order) != len(mapping) or set(order) != set(mapping):
            raise ValueError("工具展示顺序必须恰好包含八个工具")
        tools = [mapping[name] for name in order]
    return tools


def schema_hash():
    return digest(list(_SCHEMAS))


def validate_arguments(name, arguments):
    spec = next((t["function"]["parameters"] for t in _SCHEMAS if t["function"]["name"] == name), None)
    if spec is None or not isinstance(arguments, dict):
        raise ValueError("未知工具或参数不是对象")
    if set(arguments) - spec["properties"].keys() or set(spec["required"]) - arguments.keys():
        raise ValueError("缺少必填参数或含有未声明参数")
    for key, value in arguments.items():
        rule = spec["properties"][key]
        kind = rule["type"]
        if kind == "string":
            if not isinstance(value, str) or not rule["minLength"] <= len(value) <= rule["maxLength"]:
                raise ValueError("字符串参数类型或长度不合法: " + key)
            if rule.get("format") == "date-time":
                stamp(value)
            if rule.get("format") == "date":
                if datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") != value:
                    raise ValueError("日期必须为 YYYY-MM-DD")
        elif kind == "integer":
            if type(value) is not int or not rule["minimum"] <= value <= rule["maximum"]:
                raise ValueError("整数参数类型或范围不合法: " + key)
        elif kind == "boolean" and type(value) is not bool:
            raise ValueError("布尔参数类型不合法: " + key)
    if name == "list_events" and stamp(arguments["end_time"]) <= stamp(arguments["start_time"]):
        raise ValueError("查询结束时间必须晚于开始时间")
    if name == "update_event" and not (set(arguments) - {"event_id"}):
        raise ValueError("至少提供一个待修改字段")


class ToolError(Exception):
    def __init__(self, code, **data):
        super().__init__(code)
        self.code, self.data = code, data


def success(data):
    return {"status": "success", "data": copy.deepcopy(data)}


def error(code, **data):
    return {"status": "error", "error_code": code, "data": copy.deepcopy(data)}
