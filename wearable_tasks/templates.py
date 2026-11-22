"""Hand-authored task families. Structured specifications precede Chinese rendering."""
from __future__ import annotations

import copy
import random
from datetime import datetime, timedelta

from . import SCHEMA_VERSION
from .contract import TZ, digest, stamp, tool_schemas

BASE_KINDS = (
    "weather_query", "events_query", "create", "existing_reminder", "create_reminder",
    "update", "delete", "weather_activity", "preferences_activity", "create_reschedule",
    "clarify_time", "clarify_duration", "retry_weather", "conflict_fallback", "create_cancel",
)
LEVELS = {"weather_query": 1, "events_query": 1, "create": 1, "existing_reminder": 2,
          "create_reminder": 2, "update": 2, "delete": 2, "weather_activity": 3,
          "preferences_activity": 3, "clarify_time": 4, "clarify_duration": 4,
          "retry_weather": 5, "conflict_fallback": 5, "create_reschedule": 6,
          "create_cancel": 6, "weather_reschedule_remind": 6}
HELD_OUT_COMPOSITION = "weather>create>followup_update>reminder"
ROBUST_KINDS = ("retry_weather", "conflict_fallback", "create_reschedule", "create_cancel")


def activities():
    return [
        {"activity_id": "run", "title": "户外跑步", "indoor": False, "duration_minutes": 30},
        {"activity_id": "walk", "title": "户外散步", "indoor": False, "duration_minutes": 20},
        {"activity_id": "stretch", "title": "室内拉伸", "indoor": True, "duration_minutes": 20},
        {"activity_id": "reading", "title": "室内阅读", "indoor": True, "duration_minutes": 30},
        {"activity_id": "meeting", "title": "事项沟通", "indoor": True, "duration_minutes": 30},
    ]


def shape_for(ordinal):
    # At least two events: spacing is an actual relationship, never unused metadata.
    count = 2 + ordinal % 5
    return {"count": count, "spacing": 45 + 15*((ordinal//5) % 5),
            "duration": 10 + 5*((ordinal//25) % 5), "anchor": 480 + 10*((ordinal//125) % 4),
            "target_index": (ordinal//5) % count, "reminder_pattern": ordinal % 3}


def plan_groups(counts, seed):
    if list(counts) != ["train", "val", "test_id", "test_composition", "test_robustness"]:
        raise ValueError("配置必须按五种 split 的固定顺序声明")
    if any(type(n) is not int or n <= 0 or n % 4 for n in counts.values()):
        raise ValueError("每个 split 的数量必须是正的四的倍数，以完整保留反事实/复述组")
    ordinals, result, cursor = {}, [], 0
    # Split allocation is performed here, before any parameter/text variant exists.
    for split, count in counts.items():
        for index in range(count//4):
            if split == "test_composition":
                kind = "weather_reschedule_remind"
            elif split == "test_robustness":
                kind = ROBUST_KINDS[index % len(ROBUST_KINDS)]
            else:
                kind = BASE_KINDS[cursor % len(BASE_KINDS)]
                cursor += 1
            ordinal = ordinals.get(kind, 0)
            ordinals[kind] = ordinal + 1
            semantic = {"kind": kind, "world_shape": shape_for(ordinal)}
            result.append({"group_id": digest(semantic), "split": split, "semantic_spec": semantic,
                           "seed": seed, "calendar_index": len(result), "robust": split == "test_robustness"})
    if len({g["group_id"] for g in result}) != len(result):
        raise ValueError("语义组构造发生碰撞")
    return result


def _event(event_id, activity_id, title, start, duration):
    return {"event_id": event_id, "activity_id": activity_id, "title": title, "start_time": start.isoformat(),
            "end_time": (start+timedelta(minutes=duration)).isoformat(), "duration_minutes": duration}


def _reminder(event, minutes, reminder_id=None):
    value = {"event_id": event["event_id"], "minutes_before": minutes,
             "trigger_time": (stamp(event["start_time"])-timedelta(minutes=minutes)).isoformat()}
    if reminder_id is not None:
        value["reminder_id"] = reminder_id
    return value


def build_case(group, variant):
    semantic = group["semantic_spec"]
    kind, shape = semantic["kind"], semantic["world_shape"]
    conditional = kind in {"weather_activity", "weather_reschedule_remind"}
    paraphrase = variant//2 if conditional else variant % 2
    param_variant = 0 if conditional else variant//2
    rng = random.Random(int(digest([group["seed"], group["group_id"]])[:16], 16))
    anchors = ("2026-11-16", "2026-01-30", "2026-12-30", "2026-12-31")
    now = datetime.fromisoformat(anchors[group["calendar_index"] % len(anchors)]).replace(hour=9, tzinfo=TZ) + timedelta(days=param_variant, seconds=group["calendar_index"])
    offset = 1 + group["calendar_index"] % 2
    day = (now+timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
    relative = "明天" if offset == 1 else "后天"
    city = ("杭州", "南京", "成都", "上海")[(rng.randrange(4)+param_variant) % 4]
    start = day.replace(hour=18)
    duration = (20, 30, 40)[(rng.randrange(3)+param_variant) % 3]
    minutes = (5, 10, 15)[rng.randrange(3)]
    new_start = start+timedelta(hours=2)
    catalog = activities()
    weather = "rain" if (variant % 2 if conditional else rng.randrange(2)) else "sunny"
    initial_events = [_event(f"event_{i+1:05d}", "meeting", "同名事项", day+timedelta(minutes=shape["anchor"]+i*shape["spacing"]), shape["duration"]) for i in range(shape["count"])]
    target = initial_events[shape["target_index"]]
    initial_reminders = []
    for i, event in enumerate(initial_events):
        include = (i+shape["reminder_pattern"]) % 2 == 0
        if event["event_id"] == target["event_id"]:
            include = kind in {"update", "delete"}
        if include:
            initial_reminders.append(_reminder(event, 5, f"reminder_{i+1:05d}"))
    preferences = {"indoor": bool(rng.randrange(2)), "max_duration_minutes": (20, 30)[rng.randrange(2)]}
    state = {"weather": {city+"|"+day.date().isoformat(): weather}, "activities": catalog,
             "preferences": preferences, "events": initial_events, "reminders": initial_reminders,
             "next_event_id": 100, "next_reminder_id": 100}
    window = {"start_time": day.replace(hour=7).isoformat(), "end_time": day.replace(hour=23).isoformat()}
    intent = {"kind": kind, "city": city, "date": day.date().isoformat(), "window": window,
              "activity_id": "run", "start_time": start.isoformat(), "duration_minutes": duration,
              "minutes_before": minutes, "target_selector": {"title": target["title"], "start_time": target["start_time"]},
              "weather_options": {"rain": "stretch", "otherwise": "run"}}
    simulator = {"answers": {}, "followup": None}
    faults = []
    goal = {"kind": kind, "expected_events": copy.deepcopy(initial_events), "expected_reminders": copy.deepcopy(initial_reminders),
            "create_spec": None, "update_event_id": None, "delete_event_id": None,
            "required_queries": [], "required_asks": [], "reminder_target": None,
            "requires_followup": False, "query": None}
    target_label = target["start_time"] + " 开始的“同名事项”"
    creation = f"{relative}18:00安排{duration}分钟户外跑步"
    if kind in {"weather_query", "retry_weather"}:
        request = (f"查询{city}{relative}的天气。", f"请告诉我{relative}{city}的天气情况。")[paraphrase]
        args = {"city": city, "date": day.date().isoformat()}
        goal["query"] = {"name": "get_weather", "arguments": args}
        if kind == "retry_weather":
            request += "如果只读查询暂时不可用，请重试一次。"
            faults = [{"tool": "get_weather", "match": args, "remaining": 1}]
    elif kind == "events_query":
        request = (f"列出{relative}07:00到23:00的日程。", f"请查询{relative}07:00–23:00之间的安排。")[paraphrase]
        goal["query"] = {"name": "list_events", "arguments": window}
    elif kind in {"existing_reminder", "update", "delete"}:
        prefix = (f"请先查找{target_label}，", f"查询{relative}的安排后，针对{target_label}，")[paraphrase]
        if kind == "existing_reminder":
            request = prefix+f"提前{minutes}分钟提醒我。"
            goal["reminder_target"] = target["event_id"]
            goal["expected_reminders"] = [r for r in goal["expected_reminders"] if r["event_id"] != target["event_id"]] + [_reminder(target, minutes)]
        elif kind == "update":
            request = prefix+f"只把开始时间改为{relative}18:00，其他要求不变。"
            goal["update_event_id"] = target["event_id"]
            modified = copy.deepcopy(target)
            modified.update(start_time=start.isoformat(), end_time=(start+timedelta(minutes=target["duration_minutes"])).isoformat())
            goal["expected_events"] = [modified if e["event_id"] == target["event_id"] else e for e in goal["expected_events"]]
            for reminder in goal["expected_reminders"]:
                if reminder["event_id"] == target["event_id"]:
                    reminder["trigger_time"] = _reminder(modified, reminder["minutes_before"])["trigger_time"]
        else:
            request = prefix+"删除这项日程及其关联提醒，其他安排保留。"
            goal["delete_event_id"] = target["event_id"]
            goal["expected_events"] = [e for e in goal["expected_events"] if e["event_id"] != target["event_id"]]
            goal["expected_reminders"] = [r for r in goal["expected_reminders"] if r["event_id"] != target["event_id"]]
        goal["required_queries"].append({"name": "list_events", "contains_event_id": target["event_id"]})
    elif kind == "weather_activity" or kind == "weather_reschedule_remind":
        request = (f"先查询{city}{relative}天气；下雨就选室内拉伸，否则选户外跑步，{relative}18:00安排{duration}分钟。",
                   f"请根据{city}{relative}的天气安排活动：有雨用室内拉伸，无雨用户外跑步；开始于{relative}18:00，持续{duration}分钟。")[paraphrase]
        intent["activity_id"] = None
        goal["required_queries"].append({"name": "get_weather", "arguments": {"city": city, "date": day.date().isoformat()}})
    elif kind == "preferences_activity":
        request = (f"查询我的室内/室外偏好和活动目录，选符合偏好且不超过偏好时长的最短活动（并列按ID排序），{relative}18:00按目录时长安排。",
                   f"请按已保存的室内属性与最长时长筛选目录，选时长最短的一项，并列选ID最小的，在{relative}18:00安排，时长用目录值。")[paraphrase]
        intent["activity_id"], intent["duration_minutes"] = None, None
        goal["required_queries"].extend([{"name": "get_preferences"}, {"name": "search_activity", "selected_activity": True}])
    else:
        request = ("请"+creation+"。", "帮我"+creation+"，其他安排保留。")[paraphrase]
        if kind == "create_reminder":
            request += f"并提前{minutes}分钟提醒。"
        elif kind == "clarify_time":
            request = (f"请安排{duration}分钟户外跑步，开始时间请先问我。", f"帮我建一项{duration}分钟户外跑步；我还没提供开始时间，请先询问。")[paraphrase]
            simulator["answers"] = {"start_time": start.isoformat()}
            intent["start_time"] = None
            goal["required_asks"] = ["start_time"]
        elif kind == "clarify_duration":
            request = (f"请在{relative}18:00安排户外跑步，持续时长请先问我。", f"帮我安排{relative}18:00的户外跑步；时长尚未确定，请询问我。")[paraphrase]
            simulator["answers"] = {"duration_minutes": duration}
            intent["duration_minutes"] = None
            goal["required_asks"] = ["duration_minutes"]
        elif kind == "conflict_fallback":
            occupied = _event("busy_original", "meeting", "已有会议", start, duration)
            state["events"].append(occupied)
            goal["expected_events"].append(copy.deepcopy(occupied))
            alt1, alt2 = start+timedelta(hours=1), start+timedelta(hours=2)
            if shape["count"] % 2 == 0:
                other = _event("busy_alternative", "meeting", "备选时段会议", alt1, duration)
                state["events"].append(other)
                goal["expected_events"].append(copy.deepcopy(other))
            intent["alternatives"] = [alt1.isoformat(), alt2.isoformat()]
            request += f"如果冲突，只允许依次改到{relative}19:00、20:00，选第一个可用时间；不动已有日程。"
    creates = kind not in {"weather_query", "retry_weather", "events_query", "existing_reminder", "update", "delete"}
    chosen_activity, chosen_duration, chosen_start = "run", duration, start
    if conditional:
        chosen_activity = "stretch" if weather == "rain" else "run"
    if kind == "preferences_activity":
        candidates = [a for a in catalog if a["indoor"] == preferences["indoor"] and a["duration_minutes"] <= preferences["max_duration_minutes"]]
        selected = min(candidates, key=lambda a: (a["duration_minutes"], a["activity_id"]))
        chosen_activity, chosen_duration = selected["activity_id"], selected["duration_minutes"]
    if kind == "conflict_fallback":
        chosen_start = start+timedelta(hours=2 if shape["count"] % 2 == 0 else 1)
    if creates:
        activity = next(a for a in catalog if a["activity_id"] == chosen_activity)
        created = _event("$created", chosen_activity, activity["title"], chosen_start, chosen_duration)
        goal["create_spec"] = {k: created[k] for k in ("activity_id", "start_time", "duration_minutes")}
        if kind != "create_cancel":
            goal["expected_events"].append(copy.deepcopy(created))
        if kind == "create_reminder":
            goal["reminder_target"] = "$created"
            goal["expected_reminders"].append(_reminder(created, minutes))
        if kind in {"create_reschedule", "weather_reschedule_remind", "create_cancel"}:
            goal["requires_followup"] = True
            if kind == "create_cancel":
                reply = {"content": "我改主意了，请撤销刚创建的活动，其他安排不变。", "slots": {"intent": "cancel"}}
                goal["delete_event_id"] = "$created"
            else:
                reply = {"content": f"请只把刚创建活动的开始时间改为{new_start.isoformat()}，时长和活动不变。",
                         "slots": {"intent": "reschedule", "start_time": new_start.isoformat()}}
                goal["update_event_id"] = "$created"
                goal["expected_events"][-1].update(start_time=new_start.isoformat(), end_time=(new_start+timedelta(minutes=chosen_duration)).isoformat())
                if kind == "weather_reschedule_remind":
                    reply["content"] += f"然后提前{minutes}分钟提醒我。"
                    reply["slots"]["minutes_before"] = minutes
                    goal["reminder_target"] = "$created"
                    goal["expected_reminders"].append(_reminder(goal["expected_events"][-1], minutes))
            simulator["followup"] = {"trigger": "after_create_success", "reply": reply}
    tool_order = [t["function"]["name"] for t in tool_schemas()]
    if group["robust"]:
        random.Random(int(digest([group["group_id"], variant, "tool-order"])[:16], 16)).shuffle(tool_order)
        canonical_order = [t["function"]["name"] for t in tool_schemas()]
        if tool_order == canonical_order:
            tool_order = tool_order[1:]+tool_order[:1]
    request = "以模拟当前时间 " + now.isoformat() + " 为准，" + request
    return {"schema_version": SCHEMA_VERSION, "case_id": digest([group["group_id"], variant, group["seed"], group["calendar_index"], group["split"]])[:24],
            "group_id": group["group_id"], "template_id": kind+"_zh_"+str(paraphrase), "split": group["split"], "level": LEVELS[kind],
            "visible_context": {"request": request, "now": now.isoformat(), "timezone": "Asia/Shanghai"},
            "initial_state": state, "goal": goal, "user_simulator": simulator, "fault_plan": faults,
            "max_tool_calls": 20, "reference_intent": intent, "semantic_spec": copy.deepcopy(semantic),
            "composition_tag": HELD_OUT_COMPOSITION if kind == "weather_reschedule_remind" else kind,
            "tool_order": tool_order, "branch": weather if conditional else "not_conditional"}
