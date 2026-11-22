import pytest

from lightagent.cases import REQUEST, cases, scenario
from lightagent.environment import Environment
from lightagent.evaluation import assess
from lightagent.models import Call, Final
from lightagent.policy import RulePolicy
from lightagent.runtime import Runtime


@pytest.mark.parametrize("case", list(cases().values()), ids=lambda c: c.name)
def test_complete_executable_scenarios(case):
    runtime = Runtime(Environment(case.scenario), RulePolicy())
    runtime.run(case.request)
    for reply in case.replies:
        assert runtime.last_final.status == "clarification"
        runtime.run(reply)
    result = assess(case, runtime)
    assert result["passed"], result


class ScriptedPolicy:
    label = "adversarial-test"

    def __init__(self, *actions):
        self.actions = iter(actions)

    def next(self, messages, tools):
        return next(self.actions)


def test_fabricated_success_is_blocked():
    runtime = Runtime(Environment(scenario()), ScriptedPolicy(Final(status="success", message="已经安排好", event_id="fake", reminder_id="fake")))
    final = runtime.run(REQUEST)
    assert final.status == "failed"
    assert "拦截" in final.message
    assert not runtime.env.events


def test_step_budget_discloses_partial_write():
    runtime = Runtime(Environment(scenario()), RulePolicy(), max_steps=5)
    final = runtime.run(REQUEST)
    assert len(runtime.env.events) == 1
    assert not runtime.env.reminders
    assert final.status == "partial"
    assert "上限" in final.message


def test_non_success_free_text_cannot_claim_success():
    runtime = Runtime(Environment(scenario()), ScriptedPolicy(Final(status="failed", message="已成功创建所有日程和提醒")))
    final = runtime.run(REQUEST)
    assert "已成功" not in final.message
    assert "未创建" in final.message


def test_new_task_does_not_claim_previous_task_writes():
    env = Environment(scenario())
    first = Runtime(env, RulePolicy())
    assert first.run(REQUEST).status == "success"
    second = Runtime(env, ScriptedPolicy(Final(status="success", message="完成", event_id=first.last_final.event_id, reminder_id=first.last_final.reminder_id)))
    final = second.run("另一个任务")
    assert final.status == "failed"
    assert second.task_events() == []


def test_completed_session_requires_new_runtime():
    runtime = Runtime(Environment(scenario()), RulePolicy())
    runtime.run(REQUEST)
    with pytest.raises(ValueError, match="任务已结束"):
        runtime.run(REQUEST)


def test_unsupported_intent_never_creates_exercise():
    runtime = Runtime(Environment(scenario()), RulePolicy())
    final = runtime.run("帮我删除所有会议")
    assert final.status == "clarification"
    assert runtime.env.trace == []


@pytest.mark.parametrize("user_text", [
    REQUEST.replace("户外跑步；否则安排室内拉伸", "室内拉伸；否则安排户外跑步"),
    REQUEST.replace("就安排半小时", "就不要安排半小时"),
    REQUEST.replace("如果没有安排", "如果已有安排"),
    "取消任务。" + REQUEST,
])
def test_rule_grammar_rejects_changed_conditions_and_negative_instructions(user_text):
    runtime = Runtime(Environment(scenario()), RulePolicy())
    assert runtime.run(user_text).status == "clarification"
    assert runtime.env.trace == []


def test_evaluator_detects_wrong_state_even_when_status_is_success():
    case = cases()["sunny"]
    runtime = Runtime(Environment(case.scenario), RulePolicy())
    runtime.run(case.request)
    runtime.env.events[0]["activity"] = "indoor_stretch"
    result = assess(case, runtime)
    assert "wrong_branch_or_city" in result["failures"]
    assert not result["task_completed"]


def test_unattempted_failure_is_not_a_correct_failure_report():
    case = cases()["event_failure"]
    runtime = Runtime(Environment(case.scenario), ScriptedPolicy(Final(status="failed", message="做不了")))
    runtime.run(case.request)
    result = assess(case, runtime)
    assert "missing_expected_failure_evidence" in result["failures"]

