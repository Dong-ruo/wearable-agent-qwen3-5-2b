"""Hand-written integration scenarios. These are checks, not a training corpus."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import Scenario


REQUEST = "帮我检查明天下午六点是否有空。如果没有安排，而且杭州不下雨，就安排半小时户外跑步；否则安排室内拉伸，提前十分钟提醒我。"


@dataclass
class Case:
    name: str
    description: str
    scenario: Scenario
    request: str = REQUEST
    replies: list[str] = field(default_factory=list)
    expected_status: str = "success"
    expected_activity: str | None = "outdoor_run"
    expected_start: str = "2026-11-17T18:00:00+08:00"
    duration_minutes: int = 30
    reminder_minutes: int = 10
    city: str = "杭州"
    expected_event_count: int = 1
    expected_reminder_count: int = 1
    requires_consent: bool = False


def scenario(name="sunny", **overrides):
    fields = {"name": name, "now": "2026-11-16T10:00:00+08:00", "timezone": "Asia/Shanghai",
              "weather": {"杭州|2026-11-17": "sunny"}}
    fields.update(overrides)
    return Scenario.model_validate(fields)


def cases() -> dict[str, Case]:
    meeting = {"id": "meeting-1", "title": "原有会议", "start": "2026-11-17T18:00:00+08:00",
               "end": "2026-11-17T19:00:00+08:00", "origin": "external"}
    result = [
        Case("sunny", "空闲且晴天：户外跑步", scenario()),
        Case("rain", "空闲但下雨：室内拉伸", scenario("rain", weather={"杭州|2026-11-17": "rain"}), expected_activity="indoor_stretch"),
        Case("cloudy", "多云不下雨：户外跑步", scenario("cloudy", weather={"杭州|2026-11-17": "cloudy"})),
        Case("busy", "已有会议：查询空档并请求顺延许可", scenario("busy", events=[meeting]),
             expected_status="clarification", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("busy_confirmed", "用户同意后顺延，不覆盖会议", scenario("busy_confirmed", events=[meeting]),
             replies=["同意顺延"], expected_activity="indoor_stretch", expected_start="2026-11-17T19:00:00+08:00", requires_consent=True),
        Case("busy_declined", "用户取消后不写入", scenario("busy_declined", events=[meeting]),
             replies=["取消"], expected_status="failed", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("race", "查询后出现临时会议：重新找空档并询问", scenario("race", faults={"race_before_create": True}),
             expected_status="clarification", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("race_confirmed", "临时冲突后用户同意顺延", scenario("race_confirmed", faults={"race_before_create": True}),
             replies=["同意顺延"], expected_activity="indoor_stretch", expected_start="2026-11-17T18:30:00+08:00", requires_consent=True),
        Case("event_failure", "日程写入失败不得报成功", scenario("event_failure", faults={"event_create_failures": 1}),
             expected_status="failed", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("reminder_failure", "日程成功但提醒失败，报告部分完成", scenario("reminder_failure", faults={"reminder_create_failures": 1}),
             expected_status="partial", expected_reminder_count=0),
        Case("verify_failure", "日程回读失败，不得报全部完成", scenario("verify_failure", faults={"unavailable": ["calendar_get"]}),
             expected_status="partial", expected_reminder_count=0),
        Case("reminder_verify_failure", "提醒已写入但回读失败", scenario("reminder_verify_failure", faults={"unavailable": ["reminder_get"]}),
             expected_status="partial"),
        Case("weather_missing", "天气缺失不能默认为晴天", scenario("weather_missing", weather={}),
             expected_status="failed", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("calendar_unavailable", "日程查询失败时停止写入", scenario("calendar_unavailable", faults={"unavailable": ["calendar_list"]}),
             expected_status="failed", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("missing_city", "缺少城市先询问", scenario("missing_city"), request=REQUEST.replace("杭州", ""),
             expected_status="clarification", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("city_followup", "用户补充城市后继续执行", scenario("city_followup"), request=REQUEST.replace("杭州", ""), replies=["城市是杭州"]),
        Case("ambiguous_time", "未说明上午/下午先询问", scenario("ambiguous_time"), request=REQUEST.replace("下午", ""),
             expected_status="clarification", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("time_followup", "补充明确时间后继续执行", scenario("time_followup"), request=REQUEST.replace("下午", ""), replies=["明天下午六点"]),
        Case("no_slot", "两小时内无空档时准确反馈", scenario("no_slot", events=[{**meeting, "end": "2026-11-17T21:00:00+08:00"}]),
             expected_status="failed", expected_activity=None, expected_event_count=0, expected_reminder_count=0),
        Case("midnight", "跨日：按当地时区解析明天", scenario("midnight", now="2026-11-16T16:30:00+00:00", weather={"杭州|2026-11-18": "sunny"}),
             expected_start="2026-11-18T18:00:00+08:00"),
    ]
    return {c.name: c for c in result}

