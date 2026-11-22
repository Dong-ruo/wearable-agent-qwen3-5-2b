"""Public exports are built from actual replayed messages, not task/goal serialization."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from . import TOOL_VERSION
from .contract import canonical, tool_schemas
from .reference import replay

FORBIDDEN = frozenset({"goal", "initial_state", "fault_plan", "reference_intent", "user_simulator",
                       "semantic_spec", "group_id", "template_id", "split", "level", "validation"})


def assert_public(value):
    if isinstance(value, dict):
        if FORBIDDEN & value.keys():
            raise ValueError("公开结构包含私有字段: " + str(sorted(FORBIDDEN & value.keys())))
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)
    elif isinstance(value, str):
        # Tool/final content is JSON text; inspect structured content too when parseable.
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return
        if isinstance(parsed, (dict, list)):
            assert_public(parsed)


def sft_record(task, trajectory, verified_env=None):
    env = verified_env if verified_env is not None else replay(task, trajectory)
    if not env.evaluate()["success"] or env._task["case_id"] != task["case_id"]:
        raise ValueError("仅允许导出已经真实执行且验证成功的轨迹")
    # Copy only messages produced by the executor; never use an unverified supplied transcript.
    record = {"messages": env.observation()["messages"], "tools": tool_schemas(task.get("tool_order"))}
    assert_public(record)
    return record


def grpo_reference(task):
    observation = {"messages": [
        {"role": "system", "content": "本地穿戴助手。模拟当前时间：" + task["visible_context"]["now"] + "，时区 Asia/Shanghai。所有工具只操作本地模拟状态。"},
        {"role": "user", "content": task["visible_context"]["request"]}],
        "tools": tool_schemas(task.get("tool_order"))}
    assert_public(observation)
    return {"case_id": task["case_id"], "tool_version": TOOL_VERSION, **observation}


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical(record)+"\n")
    temporary.replace(path)


class TaskStore:
    """Private store lookup. A case_id reference is never expanded into model messages."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self._index = None

    def _load(self):
        report = json.loads((self.directory / "validation-report.json").read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise ValueError("拒绝加载未通过完整验证的任务库")
        index = {}
        for split in ("train", "val", "test_id", "test_composition", "test_robustness"):
            path = self.directory / "private" / (split+".jsonl")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    task = json.loads(line)
                    if task["case_id"] in index or task["split"] != split:
                        raise ValueError("私有任务库 ID 重复或 split 不一致")
                    index[task["case_id"]] = task
        self._index = index

    def get(self, case_id):
        if self._index is None:
            self._load()
        if case_id not in self._index:
            raise KeyError("未知 case_id")
        return copy.deepcopy(self._index[case_id])


class ReferenceEnvironmentLoader:
    """Framework-neutral reset/step/reward API; NOT a ready-made TRL Trainer adapter."""
    def __init__(self, directory):
        self.store = TaskStore(directory)
        self.env = None

    def reset(self, case_id):
        from .environment import TaskEnvironment
        self.env = TaskEnvironment(self.store.get(case_id))
        return self.env.observation()

    def step(self, action):
        if self.env is None:
            raise ValueError("先 reset(case_id)")
        return self.env.step(action)

    def terminal_reward(self):
        if self.env is None or not self.env._done:
            raise ValueError("尚未正常终止")
        return self.env.evaluate()["reward"]
