"""Independent turn scoring through the same SessionManager used by the web UI."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import mean, median

from wearable_tasks.contract import WRITES
from .evaluation import sha, write_json
from .service import SessionManager

VARIANTS = {"baseline": ("baseline", False), "memory": ("state", False),
            "full": ("full", False), "fast": ("full", True)}


def load_cases(path, case_ids=None):
    cases = json.loads(Path(path).read_text(encoding="utf-8-sig"))["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate interactive evaluation case ID")
    selected = [case for case in cases if not case_ids or case["id"] in case_ids]
    if not selected or (case_ids and set(case_ids) != {case["id"] for case in selected}):
        raise ValueError("Unknown or empty interactive evaluation selection")
    for case in selected:
        if not case["id"].isascii() or not case["id"].replace("-", "").isalnum() or not case["turns"]:
            raise ValueError("Invalid interactive evaluation case")
        for turn in case["turns"]:
            if not turn["text"].strip() or turn["expect"]["outcome"] not in {"completed", "clarified", "blocked", "detour"}:
                raise ValueError("Invalid interactive evaluation turn")
            repeat = turn.get("repeat", 1)
            if type(repeat) is not int or not 1 <= repeat <= 30 or (repeat > 1 and turn["expect"]["outcome"] != "detour"):
                raise ValueError("Only detours can repeat, up to 30 times")
    return selected


def _contains(value, expected):
    if isinstance(expected, dict):
        return isinstance(value, dict) and all(key in value and _contains(value[key], item) for key, item in expected.items())
    return value == expected


def _score_turn(expected, session, offset, timeline_offset, usage_offset):
    entries = session.tool_events[offset:]
    events = session.timeline[timeline_offset:]
    writes = [entry for entry in entries if entry["name"] in WRITES and entry["result"]["status"] == "success"]
    wants = expected.get("writes", [])
    errors = sum(index >= len(wants) or not _contains(entry, wants[index]) for index, entry in enumerate(writes))
    checks = {"write_trace": errors == 0 and len(writes) == len(wants)}
    for event_id, bundle in expected.get("objects", {}).items():
        checks[f"object:{event_id}"] = _contains(session.backend.read_event_bundle(event_id), bundle)
    for name, expected_result in expected.get("reads", {}).items():
        checks[f"read:{name}"] = any(entry["name"] == name and _contains(entry, expected_result) for entry in entries)
    if "constraints" in expected:
        checks["constraint_state"] = _contains(session.state["task_constraints"], expected["constraints"])
    evidence = all(checks.values())
    final = session.final or {}
    claims = final.get("status") == "success"
    final_correct = _contains(final.get("result"), expected.get("final_result", {})) if "final_result" in expected else True
    outcome = expected["outcome"]
    if outcome == "completed":
        passed = evidence and claims and final_correct
    elif outcome == "clarified":
        passed = evidence and session.state["phase"] == "waiting_for_user" and set(session.state["waiting_slots"]) == set(expected["slots"])
    elif outcome == "blocked":
        passed = (evidence and final.get("status") == "failed"
                  and ("error_code" not in expected or (final.get("result") or {}).get("error_code") == expected["error_code"]))
    else:
        passed = evidence
    measured = session.usages[usage_offset:]
    model_calls = sum(e["type"] == "model_start" for e in events)
    complete_usage = len(measured) == model_calls and all(type(u.get("prompt_tokens")) is int for u in measured)
    raw = session.raw_final or {}
    raw_correct = outcome in {"completed", "detour"} and evidence
    if "final_result" in expected:
        raw_correct = raw_correct and _contains(raw.get("result"), expected["final_result"])
    return {"expected": outcome, "passed": passed, "checks": checks, "actual_phase": session.state["phase"], "final": final,
            "wrong_writes": errors, "accepted_success": claims,
            "false_success": claims and (not evidence or not final_correct or outcome in {"clarified", "blocked"}),
            "model_claimed_success": raw.get("status") == "success",
            "raw_false_success": raw.get("status") == "success" and not raw_correct,
            "valid_execution": outcome == "completed" and evidence,
            "overblock": (outcome == "completed" and evidence and raw.get("status") == "success"
                          and ("final_result" not in expected or _contains(raw.get("result"), expected["final_result"]))
                          and any(e["type"] == "final_block" for e in events)),
            "model_calls": model_calls, "input_tokens": sum(u["prompt_tokens"] for u in measured) if complete_usage else None}


def _summary(cases):
    turns = [turn for case in cases for turn in case["turns"] if turn["expected"] != "detour"]

    def rate(numerator, denominator):
        return {"count": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}

    claims = sum(t["accepted_success"] for t in turns)
    raw_claims = sum(t["model_claimed_success"] for t in turns)
    result = {"cases_passed": rate(sum(c["passed"] for c in cases), len(cases)),
              "false_success": rate(sum(t["false_success"] for t in turns), claims),
              "raw_false_success": rate(sum(t["raw_false_success"] for t in turns), raw_claims),
              "overblock": rate(sum(t["overblock"] for t in turns), sum(t["valid_execution"] for t in turns)),
              "wrong_writes": sum(t["wrong_writes"] for case in cases for t in case["turns"]),
              "mean_model_calls_per_case": mean(c["model_calls"] for c in cases),
              "median_wall_seconds": median(c["wall_seconds"] for c in cases)}
    for outcome in ("completed", "clarified", "blocked"):
        relevant = [turn for turn in turns if turn["expected"] == outcome]
        result[outcome] = rate(sum(t["passed"] for t in relevant), len(relevant))
    tokens = [c["input_tokens"] for c in cases if c["input_tokens"] is not None]
    result["input_tokens_measured_cases"] = len(tokens)
    result["mean_input_tokens_per_measured_case"] = mean(tokens) if tokens else None
    return result


def evaluate_interactive(config, endpoint_factory, cases_file, output, variants, case_ids=None):
    from time import perf_counter

    selected = load_cases(cases_file, case_ids)
    output = Path(output)
    if config.get("tool_backend", {}).get("kind", "local") != "local":
        raise ValueError("Interactive evaluation requires the isolated local backend")
    variants = list(dict.fromkeys(variants))
    if not variants or set(variants) - VARIANTS.keys():
        raise ValueError("Invalid interactive evaluation variant")
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "interactive_web_session_manager", "status": "running",
              "cases_sha256": sha(cases_file), "world_sha256": sha(config["world_file"]),
              "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
              "config": config, "selected_cases": [case["id"] for case in selected], "variants": {}}
    write_json(output / "report.json", report)
    for variant in variants:
        profile, fast = VARIANTS[variant]
        cfg = copy.deepcopy(config)
        cfg["runtime"]["local_fast_path"] = fast
        endpoint = endpoint_factory()
        manager = SessionManager(cfg, endpoint, output / f"{variant}.sqlite3")
        rows = []
        try:
            for case in selected:
                started = perf_counter()
                offsets = (0, 0, 0)
                turns = []
                sid = None
                instructions = [turn for turn in case["turns"] for _ in range(turn.get("repeat", 1))]
                for index, instruction in enumerate(instructions):
                    request_id = f"{case['id']}:{index}"
                    if sid is None:
                        sid = manager.create(instruction["text"], request_id, profile, defer=True)["session_id"]
                        manager.run(sid, request_id + ":run")
                    else:
                        manager.message(sid, instruction["text"], request_id)
                    session = manager.store.load(sid, endpoint, manager.endpoint_hash)
                    score = _score_turn(instruction["expect"], session, *offsets)
                    turns.append(score)
                    offsets = (len(session.tool_events), len(session.timeline), len(session.usages))
                wall = perf_counter() - started
                row = {"id": case["id"], "session_id": sid, "passed": all(turn["passed"] for turn in turns),
                       "turns": turns, "model_calls": sum(turn["model_calls"] for turn in turns), "wall_seconds": wall,
                       "input_tokens": sum(turn["input_tokens"] for turn in turns) if all(turn["input_tokens"] is not None for turn in turns) else None}
                rows.append(row)
                report["variants"][variant] = {"profile": profile, "fast_path": fast, "summary": _summary(rows), "cases": rows}
                write_json(output / "report.json", report)
                print(f"{variant} {case['id']} {'pass' if row['passed'] else 'fail'} {row['model_calls']} model calls", flush=True)
        finally:
            manager.store.close()
    report["status"] = "completed"
    write_json(output / "report.json", report)
    return report
