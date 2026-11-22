"""Private task handling stays in backends, never in model-facing session state."""
from __future__ import annotations

import copy
import json
import urllib.error
import urllib.request
import os
from functools import lru_cache

from wearable_tasks import SCHEMA_VERSION
from wearable_tasks.contract import canonical, tool_schemas, validate_arguments, READ_ONLY
from wearable_tasks.environment import TaskEnvironment
from wearable_tasks.exporting import TaskStore


@lru_cache(maxsize=8)
def task_store(directory):
    return TaskStore(directory)


class TaskBackend:
    kind = "dataset"

    def __init__(self, directory, case_id):
        self.directory, self.case_id = str(directory), case_id
        self.__env = TaskEnvironment(task_store(str(directory)).get(case_id))
        self.actions, self.receipts = [], []

    def observation(self):
        return self.__env.observation()

    def step(self, action):
        result = self.__env.step(action)
        self.actions.append(copy.deepcopy(action))
        # Never expose reward, goal, state, fault plan or validation to the runner.
        receipt = {"observation": result["observation"], "done": result["done"]}
        self.receipts.append(copy.deepcopy(receipt))
        return receipt

    def offline_evaluate(self, proposed_final=None):
        if proposed_final is None: return self.__env.evaluate()
        env = copy.deepcopy(self.__env)
        env._final, env._done = copy.deepcopy(proposed_final), True
        return env.evaluate()

    def export(self):
        return {"kind": self.kind, "directory": self.directory, "case_id": self.case_id,
                "actions": copy.deepcopy(self.actions), "receipts": copy.deepcopy(self.receipts)}


class _InteractiveEngine(TaskEnvironment):
    def evaluate(self):
        return {"reward": 0, "evaluated": False}

    def read_event_bundle(self, event_id):
        event = next((row for row in self._state["events"] if row["event_id"] == event_id), None)
        reminder = next((row for row in self._state["reminders"] if row["event_id"] == event_id), None)
        return {"event": copy.deepcopy(event), "reminder": copy.deepcopy(reminder), "event_count": len(self._state["events"])}

    def activity_location(self, activity_id):
        return next((row["indoor"] for row in self._state["activities"] if row["activity_id"] == activity_id), None)

    def matching_activity_ids(self, title):
        rows = self._state["activities"]
        exact = [row["activity_id"] for row in rows if row["title"] == title]
        return exact or [row["activity_id"] for row in rows if title in row["title"]]


class LocalBackend:
    """Interactive sandbox using the frozen eight-tool engine, without hidden task goals."""
    kind = "local"

    def __init__(self, world, request):
        self.world, self.request = copy.deepcopy(world), request
        task = {"schema_version": SCHEMA_VERSION, "case_id": "interactive", "initial_state": world["state"],
                "visible_context": {"now": world["now"], "request": request}, "fault_plan": [],
                "user_simulator": {"answers": {}, "followup": None}, "max_tool_calls": world.get("max_tool_calls", 100)}
        self.__env = _InteractiveEngine(task)
        self.actions = []

    def observation(self):
        return self.__env.observation()

    def read_event_bundle(self, event_id):
        """Controller-only readback from the actual state of this local session."""
        return self.__env.read_event_bundle(event_id)

    def activity_location(self, activity_id):
        return self.__env.activity_location(activity_id)

    def matching_activity_ids(self, title):
        return self.__env.matching_activity_ids(title)

    def step(self, action):
        if action["type"] == "ask_user":
            self.actions.append(copy.deepcopy(action))
            return {"observation": {"messages": []}, "done": False, "waiting": True}
        if self.__env._done:
            self.__env._done = False
        result = self.__env.step(action)
        self.actions.append(copy.deepcopy(action))
        return {"observation": result["observation"], "done": result["done"]}

    def user_message(self, text):
        self.actions.append({"type": "external_user", "content": text})

    def export(self):
        return {"kind": self.kind, "world": self.world, "request": self.request, "actions": copy.deepcopy(self.actions)}


class HTTPBackend:
    """Eight-tool gateway adapter. No provider credentials or real calls are bundled."""
    kind = "http"

    def __init__(self, config, request):
        from urllib.parse import urlsplit
        p = urlsplit(config["base_url"])
        if p.scheme not in {"http", "https"} or not p.hostname or p.username or p.password or p.query or p.fragment:
            raise ValueError("Invalid tool gateway URL")
        self.config, self.request, self.actions, self.receipts = copy.deepcopy(config), request, [], []
        if not self.config.get("now"):
            from datetime import datetime
            from zoneinfo import ZoneInfo
            self.config["now"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")

    def observation(self):
        return {"messages": [{"role": "system", "content": "当前时间：" + self.config["now"]},
                             {"role": "user", "content": self.request}], "tools": tool_schemas(),
                "current_time": self.config["now"], "timezone": "Asia/Shanghai"}

    def step(self, action):
        import uuid
        self.actions.append(copy.deepcopy(action))
        if action["type"] == "ask_user":
            return {"observation": {"messages": []}, "done": False, "waiting": True}
        if action["type"] == "final":
            return {"observation": {"messages": []}, "done": True}
        name, args = action["name"], action["arguments"]
        try: validate_arguments(name, args)
        except ValueError:
            result = {"status": "error", "error_code": "INVALID_ARGUMENT", "data": {"source": "gateway_adapter_validation"}}
            self.receipts.append({"name": name, "arguments": copy.deepcopy(args), "result": result, "idempotency_key": None})
            return {"observation": {"messages": [], "tool_result": result}, "done": False}
        headers = {"Content-Type": "application/json", "Idempotency-Key": action.get("request_id", uuid.uuid4().hex)}
        key = os.environ.get(self.config.get("api_key_env", "TOOL_API_KEY"))
        if key: headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(self.config["base_url"].rstrip("/") + "/tools/" + name,
                                     data=canonical(args).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.config.get("timeout", 15)) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000: raise ValueError("Oversized tool response")
            from .protocol import strict_json
            result = strict_json(raw.decode("utf-8"))
            if not isinstance(result, dict) or result.get("status") not in {"success", "error"} or not isinstance(result.get("data"), dict):
                raise ValueError("Invalid tool gateway response")
            if result["status"] == "error" and not isinstance(result.get("error_code"), str):
                raise ValueError("Missing error code")
            if result["status"] == "success":
                data = result["data"]
                if name in {"list_events", "search_activity"}:
                    key = "events" if name == "list_events" else "activities"
                    identity = "event_id" if name == "list_events" else "activity_id"
                    if not isinstance(data.get(key), list) or not all(isinstance(v, dict) and isinstance(v.get(identity), str) for v in data[key]):
                        raise ValueError("Invalid tool result rows")
                    if name == "list_events" and not all(all(isinstance(v.get(k), str) for k in ("start_time", "end_time", "title", "activity_id")) and type(v.get("duration_minutes")) is int for v in data[key]):
                        raise ValueError("Incomplete event rows")
                    if name == "search_activity" and not all(isinstance(v.get("title"), str) and type(v.get("indoor")) is bool and type(v.get("duration_minutes")) is int for v in data[key]):
                        raise ValueError("Incomplete activity rows")
                if name in {"create_event", "update_event", "delete_event", "set_reminder"} and not isinstance(data.get("event_id"), str):
                    raise ValueError("Missing result event ID")
                if name in {"create_event", "update_event"} and not all(isinstance(data.get(k), str) for k in ("start_time", "end_time", "activity_id", "title")):
                    raise ValueError("Incomplete event receipt")
                if name == "get_weather" and not all(isinstance(data.get(k), str) for k in ("city", "date", "condition")):
                    raise ValueError("Incomplete weather facts")
                if name == "get_preferences" and not isinstance(data.get("preferences"), dict): raise ValueError("Invalid preferences")
                if name == "set_reminder" and (not isinstance(data.get("reminder_id"), str) or not isinstance(data.get("trigger_time"), str) or type(data.get("minutes_before")) is not int):
                    raise ValueError("Incomplete reminder receipt")
                if name == "delete_event" and (data.get("deleted") is not True or data["event_id"] != args["event_id"]):
                    raise ValueError("Invalid deletion receipt")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeError):
            result = {"status": "error", "error_code": "TEMPORARY_UNAVAILABLE" if name in READ_ONLY else "WRITE_OUTCOME_UNKNOWN", "data": {}}
        self.receipts.append({"name": name, "arguments": copy.deepcopy(args), "result": copy.deepcopy(result),
                              "idempotency_key": headers["Idempotency-Key"]})
        return {"observation": {"messages": [], "tool_result": result}, "done": False}

    def user_message(self, text):
        self.actions.append({"type": "external_user", "content": text})

    def export(self):
        return {"kind": self.kind, "config": self.config, "request": self.request,
                "actions": copy.deepcopy(self.actions), "receipts": copy.deepcopy(self.receipts)}


def restore_backend(snapshot):
    if snapshot["kind"] == "http":
        backend = HTTPBackend(snapshot["config"], snapshot["request"])
        backend.actions, backend.receipts = copy.deepcopy(snapshot["actions"]), copy.deepcopy(snapshot["receipts"])
        return backend  # Never replay remote side effects.
    backend = (TaskBackend(snapshot["directory"], snapshot["case_id"]) if snapshot["kind"] == "dataset"
               else LocalBackend(snapshot["world"], snapshot["request"]))
    for index, action in enumerate(snapshot["actions"]):
        if action["type"] == "external_user": backend.user_message(action["content"])
        else:
            actual = backend.step(action)
            if snapshot["kind"] == "dataset" and actual != snapshot["receipts"][index]:
                raise ValueError("Replay observation differs from saved receipt")
    return backend
