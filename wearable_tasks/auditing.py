from __future__ import annotations

import copy
from collections import Counter, defaultdict
from datetime import timedelta

from .contract import canonical, digest, stamp
from .environment import TaskEnvironment
from .exporting import assert_public, grpo_reference, sft_record
from .reference import replay
from .templates import HELD_OUT_COMPOSITION


def audit_splits(tasks):
    groups, fingerprints, requests, case_ids = {}, {}, {}, set()
    variants = Counter()
    conditional = defaultdict(list)
    for task in tasks:
        case_id, split, group = task["case_id"], task["split"], task["group_id"]
        if case_id in case_ids or group != digest(task["semantic_spec"]):
            raise ValueError("重复 case_id 或语义分组 ID 不符")
        case_ids.add(case_id)
        if group in groups and groups[group] != split:
            raise ValueError("group_id 跨 split 泄漏")
        groups[group] = split
        variants[group] += 1
        semantic = {key: task[key] for key in ("visible_context", "initial_state", "goal", "user_simulator", "fault_plan", "max_tool_calls", "reference_intent")}
        key = digest(semantic)
        if key in fingerprints and fingerprints[key] != split:
            raise ValueError("去元数据后出现跨 split 重复任务")
        fingerprints[key] = split
        request_key = task["visible_context"]["request"]
        if request_key in requests and requests[request_key] != split:
            raise ValueError("相同用户请求跨 split")
        requests[request_key] = split
        if task["branch"] != "not_conditional":
            conditional[(group, request_key)].append(task)
        if task["composition_tag"] == HELD_OUT_COMPOSITION and split != "test_composition":
            raise ValueError("保留组合泄漏到其他 split")
        if split == "test_composition" and task["composition_tag"] != HELD_OUT_COMPOSITION:
            raise ValueError("组合测试未使用预留组合")
    if any(count != 4 for count in variants.values()):
        raise ValueError("语义组必须完整包含四个变体")
    for pair in conditional.values():
        if len(pair) != 2 or {t["branch"] for t in pair} != {"sunny", "rain"}:
            raise ValueError("反事实配对或分支不平衡")
        choices = {t["goal"]["create_spec"]["activity_id"] for t in pair}
        if len(choices) != 2:
            raise ValueError("反事实没有改变正确活动")
    return {"passed": True, "groups": len(groups), "cross_split_groups": 0,
            "cross_split_duplicates": 0, "cross_split_duplicate_requests": 0,
            "counterfactual_pairs": len(conditional), "held_out_composition": HELD_OUT_COMPOSITION}


def audit_composition_execution(tasks, environments):
    training_names = set()
    for task, env in zip(tasks, environments):
        names = env["names"] if isinstance(env, dict) else [r["name"] for r in env._log if r["result"]["status"] == "success"]
        followup = env["followup_fired"] if isinstance(env, dict) else env._followup_fired
        if task["split"] == "train":
            training_names.update(names)
        if task["split"] == "test_composition":
            needed = ["get_weather", "create_event", "update_event", "set_reminder"]
            cursor = 0
            for name in names:
                if cursor < len(needed) and name == needed[cursor]:
                    cursor += 1
            if cursor != len(needed) or not followup:
                raise ValueError("保留组合未按依赖实际执行")
        elif "get_weather" in names and "create_event" in names and "update_event" in names and "set_reminder" in names:
            raise ValueError("实际工具依赖组合泄漏到非组合测试")
    if not {"get_weather", "create_event", "update_event", "set_reminder"} <= training_names:
        raise ValueError("保留组合使用了训练里从未见过的业务工具")
    return {"passed": True, "seen_constituent_tools": sorted(training_names)}


def audit_isolation(task):
    original = copy.deepcopy(task)
    envs = [TaskEnvironment(task) for _ in range(4)]
    baselines = [env.private_snapshot() for env in envs]
    # Mutate only one environment, through a real business tool and its own fault counter.
    if task["fault_plan"]:
        rule = task["fault_plan"][0]
        envs[0].execute(rule["tool"], rule["match"])
    else:
        envs[0].execute("create_event", {"activity_id": "run", "start_time": task["reference_intent"]["date"]+"T23:00:00+08:00", "duration_minutes": 20})
    if task != original or any(envs[i].private_snapshot() != baselines[i] for i in (1, 2, 3)):
        raise ValueError("四份 rollout 或任务对象共享可变状态")
    envs[0].reset()
    if envs[0].private_snapshot() != baselines[0]:
        raise ValueError("reset 没有完整重建状态")
    return True


def audit_privacy(task, env):
    initial = TaskEnvironment(task).observation()
    if set(initial) != {"messages", "tools", "current_time", "timezone"}:
        raise ValueError("公开 observation 不是白名单")
    assert_public(initial)
    assert_public(grpo_reference(task))
    assert_public(sft_record(task, env.trajectory(), verified_env=env))
    canary = "PRIVATE_CANARY_"+task["case_id"]
    modified = copy.deepcopy(task)
    modified["goal"]["private_canary"] = canary
    modified["reference_intent"]["private_canary"] = canary
    modified["user_simulator"]["answers"]["never_requested"] = canary
    if canary in canonical(TaskEnvironment(modified).observation()):
        raise ValueError("目标、参考信息或未询问答案泄漏")
    return True


def _replay_actions(task, actions):
    env = TaskEnvironment(task)
    for action in actions:
        env.step(copy.deepcopy(action))
    return env


def negative_variants(task, env):
    """White-box verifier attacks, never reference-policy shortcuts or training examples."""
    goal = task["goal"]
    created = next((r["result"]["data"]["event_id"] for r in env._log if r["name"] == "create_event" and r["result"]["status"] == "success"), None)
    target = created or goal["update_event_id"] or goal["reminder_target"]
    if target == "$created":
        target = created
    target_event = next((e for e in env._state["events"] if e["event_id"] == target), None)
    if target_event is not None:
        bad = copy.deepcopy(env)
        event = next(e for e in bad._state["events"] if e["event_id"] == target)
        event["activity_id"] = "walk" if event["activity_id"] != "walk" else "stretch"
        yield "wrong_activity", bad
        bad = copy.deepcopy(env)
        event = next(e for e in bad._state["events"] if e["event_id"] == target)
        event["start_time"] = (stamp(event["start_time"])+timedelta(minutes=5)).isoformat()
        event["end_time"] = (stamp(event["end_time"])+timedelta(minutes=5)).isoformat()
        yield "wrong_time", bad
    if created and target_event is not None:
        bad = copy.deepcopy(env)
        extra = copy.deepcopy(target_event)
        extra["event_id"] = "unauthorized_extra"
        bad._state["events"].append(extra)
        yield "duplicate_creation", bad
    reminder_target = created if goal["reminder_target"] == "$created" else goal["reminder_target"]
    if reminder_target is not None:
        bad = copy.deepcopy(env)
        bad._state["reminders"] = [r for r in bad._state["reminders"] if r["event_id"] != reminder_target]
        yield "missing_reminder", bad
        others = [e for e in env._state["events"] if e["event_id"] != reminder_target]
        if others:
            bad = copy.deepcopy(env)
            reminder = next(r for r in bad._state["reminders"] if r["event_id"] == reminder_target)
            reminder["event_id"] = others[0]["event_id"]
            reminder["trigger_time"] = (stamp(others[0]["start_time"])-timedelta(minutes=reminder["minutes_before"])).isoformat()
            yield "wrong_reminder_binding", bad
    mutable = {goal["update_event_id"], goal["delete_event_id"], goal["reminder_target"]}
    protected = [e for e in task["initial_state"]["events"] if e["event_id"] not in mutable]
    if protected:
        bad = copy.deepcopy(env)
        bad._state["events"] = [e for e in bad._state["events"] if e["event_id"] != protected[0]["event_id"]]
        bad._state["reminders"] = [r for r in bad._state["reminders"] if r["event_id"] != protected[0]["event_id"]]
        yield "deleted_unrelated_event", bad
    actual_actions = [row["action"] for row in env._actions]
    if goal["required_asks"]:
        without_asks = [a for a in actual_actions if a["type"] != "ask_user"]
        yield "guessed_missing_slot", _replay_actions(task, without_asks)
        postponed = [a for a in without_asks if a["type"] != "final"] + [a for a in actual_actions if a["type"] == "ask_user"] + [actual_actions[-1]]
        yield "posthoc_clarification", _replay_actions(task, postponed)
    if goal["requires_followup"]:
        bad = TaskEnvironment(task)
        for action in actual_actions:
            bad.step(action)
            if bad._followup_fired:
                event_id = next(r["result"]["data"]["event_id"] for r in bad._log if r["name"] == "create_event" and r["result"]["status"] == "success")
                bad.step({"type": "final", "status": "success", "result": {"event_id": event_id}, "message": "只处理了初次创建。"})
                break
        yield "ignored_followup", bad
    needed = {r["name"] for r in goal["required_queries"]}
    if goal["query"]:
        needed.add(goal["query"]["name"])
    if needed:
        skipped = [a for a in actual_actions if not (a["type"] == "tool_call" and a["name"] in needed)]
        yield "missing_necessary_query", _replay_actions(task, skipped)
    if goal["query"]:
        bad = copy.deepcopy(env)
        bad._final["result"] = {"fabricated": True}
        yield "fabricated_query_result", bad


def audit_negatives(task, env):
    counts, failures = Counter(), []
    for category, bad in negative_variants(task, env):
        report = bad.evaluate()
        if report["success"] or report["reward"] != 0:
            raise ValueError("判定器接受错误样本: " + category + " " + task["case_id"])
        counts[category] += 1
        failures.append({"case_id": task["case_id"], "type": category, "failure_codes": report["failure_codes"]})
    return counts, failures
