"""Resumable held-out model evaluation; scoring occurs after the public runner stops."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from wearable_tasks.contract import canonical
from . import VERSION
from .backends import TaskBackend, restore_backend
from .runtime import Session

TEST_SPLITS = ("test_id", "test_composition", "test_robustness")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != VERSION: raise ValueError("Wrong assistant configuration version")
    for key in ("endpoint", "runtime", "dataset", "world_file"): 
        if key not in config: raise ValueError("Missing configuration field: " + key)
    for name in ("max_model_calls", "max_tool_calls", "max_seconds", "max_input_chars", "max_total_tokens"):
        if type(config["runtime"].get(name)) is not int or config["runtime"][name] < 1: raise ValueError("Invalid budget: " + name)
    if config["endpoint"].get("temperature") != 0: raise ValueError("Fixed comparison uses temperature=0")
    from .protocol import ChatEndpoint
    ChatEndpoint(config["endpoint"])  # Configuration validation only; no request.
    if not isinstance(config["endpoint"].get("model"), str) or not config["endpoint"]["model"].strip(): raise ValueError("Missing model name")
    for name in ("timeout", "max_tokens"):
        if type(config["endpoint"].get(name)) is not int or config["endpoint"][name] < 1: raise ValueError("Invalid endpoint budget")
    if type(config["runtime"].get("seed")) is not int or not 0 <= config["runtime"].get("max_protocol_repairs", -1) <= 2:
        raise ValueError("Invalid seed or repair budget")
    if config.get("tool_backend", {}).get("kind", "local") not in {"local", "http"}: raise ValueError("Unsupported tool backend")
    for name in ("direct_weather_completion", "reuse_successful_reads", "grounded_assistance", "local_fast_path"):
        if name in config["runtime"] and type(config["runtime"][name]) is not bool: raise ValueError("Invalid runtime switch: " + name)
    for name in ("max_repeated_read_attempts", "max_consecutive_tool_failures"):
        if type(config["runtime"].get(name, 3)) is not int or not 1 <= config["runtime"].get(name, 3) <= 5:
            raise ValueError(name + " must be between 1 and 5")
    runtime = config["runtime"]
    if "context_window" in runtime:
        if type(runtime["context_window"]) is not int or runtime["context_window"] <= config["endpoint"]["max_tokens"]:
            raise ValueError("context_window must exceed endpoint.max_tokens")
        if not isinstance(runtime.get("context_tokenizer"), str) or not runtime["context_tokenizer"].strip():
            raise ValueError("context_tokenizer must name a local tokenizer directory")
    return config


def task_rows(config, splits):
    root = Path(config["dataset"])
    report = json.loads((root / "validation-report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed": raise ValueError("Dataset validation has not passed")
    from wearable_tasks.pipeline import source_fingerprint
    if source_fingerprint() != report["source_fingerprint"]: raise ValueError("Frozen environment implementation changed; revalidate before formal evaluation")
    rows, fingerprints = [], {}
    for split in splits:
        if split not in {"train", "val", *TEST_SPLITS}: raise ValueError("Unknown split")
        for relative in (f"public/grpo/{split}.jsonl", f"private/{split}.jsonl"):
            path = root / relative
            actual = sha(path)
            if actual != report["file_sha256"][relative]: raise ValueError("Dataset asset hash mismatch: " + relative)
            fingerprints[relative] = actual
        for line in (root / f"public/grpo/{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row["case_id"], str) or not row["case_id"].isalnum(): raise ValueError("Unsafe case ID")
            rows.append({"case_id": row["case_id"], "split": split})
    if len({r["case_id"] for r in rows}) != len(rows): raise ValueError("Duplicate public case IDs")
    return rows, fingerprints


def metrics(rows):
    failures, checks = Counter(), Counter()
    for row in rows:
        failures.update(row["validation"]["failure_codes"])
        checks.update(row["validation"]["checks"].keys())
    def applicable(prefix):
        selected = [r for r in rows if any(k.startswith(prefix) for k in r["validation"]["checks"])]
        return {"applicable": len(selected), "passed": sum(all(v for k, v in r["validation"]["checks"].items() if k.startswith(prefix)) for r in selected)}
    times = sorted(sum(t["seconds"] for t in r["snapshot"]["turns"]) for r in rows)
    usages = [u for r in rows for u in r["snapshot"]["usages"]]
    tokens = sum(u["completion_tokens"] for u in usages) if usages and all(type(u.get("completion_tokens")) is int for u in usages) else None
    return {"cases": len(rows), "successful": sum(r["validation"]["success"] for r in rows),
            "success_rate": sum(r["validation"]["success"] for r in rows) / len(rows) if rows else None,
            "failure_codes": dict(failures), "clarification": applicable("ASKED_"), "followup": applicable("FOLLOWUP_HANDLED"),
            "false_success_claims": sum(bool(r["snapshot"]["raw_final"] and r["snapshot"]["raw_final"]["status"] == "success" and not r["raw_final_validation"]["success"]) for r in rows),
            "reported_false_successes": sum(bool(r["snapshot"]["final"] and r["snapshot"]["final"]["status"] == "success" and not r["validation"]["success"]) for r in rows),
            "blocked_valid_completions": sum(bool(r["raw_final_validation"] and r["raw_final_validation"]["success"] and not r["validation"]["success"]) for r in rows),
            "raw_claim_vs_offline_score": True,
            "unauthorized_write_cases": sum(not r["validation"]["checks"].get("AUTHORIZED_WRITES", True) for r in rows),
            "successful_duplicate_create_calls": sum(max(0, sum(e["name"] == "create_event" and e["result"]["status"] == "success" for e in r["snapshot"]["tool_events"]) - 1) for r in rows),
            "protocol_errors": sum(r["snapshot"]["counters"]["protocol_errors"] for r in rows),
            "guard_blocks": sum(r["snapshot"]["counters"]["guard_blocks"] for r in rows),
            "retry_attempts": sum(r["snapshot"]["counters"]["retries"] for r in rows),
            "controller_weather_completions": sum(r["snapshot"].get("final_origin") == "controller_weather" for r in rows),
            "controller_query_completions": sum(r["snapshot"].get("final_origin") == "controller_query" for r in rows),
            "controller_verified_write_completions": sum(r["snapshot"].get("final_origin") == "controller_verified_write" for r in rows),
            "grounded_argument_repairs": sum(sum(e["type"] == "argument_repair" for e in r["snapshot"]["timeline"]) for r in rows),
            "reused_read_results": sum(r["snapshot"]["counters"].get("reused_reads", 0) for r in rows),
            "consecutive_failure_stops": sum(any(e["type"] == "failure_limit" for e in r["snapshot"]["timeline"]) for r in rows),
            "recovery_cases_successful": sum(r["validation"]["success"] and any(e["result"].get("error_code") == "TEMPORARY_UNAVAILABLE" for e in r["snapshot"]["tool_events"]) for r in rows),
            "mean_tool_calls": statistics.mean(r["snapshot"]["counters"]["tool_calls"] for r in rows) if rows else None,
            "model_seconds": sum(r["snapshot"]["counters"]["model_seconds"] for r in rows),
            "tool_seconds": sum(r["snapshot"]["counters"]["tool_seconds"] for r in rows),
            "latency_p50_seconds": statistics.median(times) if times else None,
            "latency_p95_seconds": times[max(0, math.ceil(.95 * len(times)) - 1)] if times else None,
            "completion_tokens": tokens, "ttft_seconds": None,
            "notes": "非流式无TTFT；token仅在服务端完整返回usage时统计；负例规则不是自然语言完整忠实度证明"}


def evaluate(config, endpoint, output, splits=TEST_SPLITS, profile="full", limit=None, resume=False):
    rows, assets = task_rows(config, splits)
    selected = rows[:limit] if limit else rows
    if not selected: raise ValueError("No task rows selected")
    root = Path(output)
    manifest = {"version": VERSION, "config": config, "profile": profile, "splits": list(splits),
                "limit": limit, "dataset_sha256": assets,
                "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}}
    signature = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    if root.exists():
        if not resume: raise ValueError("Output exists; use a new directory or --resume")
        old = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if old["signature"] != signature: raise ValueError("Resume model/config/data/code differs")
    else:
        root.mkdir(parents=True)
        write_json(root / "manifest.json", {**manifest, "signature": signature})
    completed = []
    for row in selected:
        path = root / "cases" / (row["case_id"] + ".json")
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("signature") != signature or result["case_id"] != row["case_id"]: raise ValueError("Case receipt does not match this run")
        else:
            backend = TaskBackend(config["dataset"], row["case_id"])
            session = Session(backend, endpoint, config["runtime"], profile)
            session.run()
            # The hidden evaluator runs strictly AFTER all model/controller decisions.
            result = {**row, "signature": signature, "validation": backend.offline_evaluate(),
                      "raw_final_validation": backend.offline_evaluate(session.raw_final) if session.raw_final else None,
                      "snapshot": session.snapshot()}
            write_json(path, result)
        completed.append(result)
        report = {"status": "running", "profile": profile, "signature": signature, "completed": len(completed), "requested": len(selected),
                  "scope": "formal_held_out" if limit is None and set(splits) == set(TEST_SPLITS) else "subset_or_development",
                  "overall": metrics(completed), "by_split": {s: metrics([r for r in completed if r["split"] == s]) for s in splits}}
        write_json(root / "report.json", report)
        print(f"{profile}: {len(completed)}/{len(selected)} {row['case_id']}", flush=True)
    if task_rows(config, splits)[1] != assets:
        report["status"] = "invalid_dataset_changed"
        write_json(root / "report.json", report)
        raise ValueError("Dataset changed during evaluation")
    report["status"] = "completed"
    write_json(root / "report.json", report)
    return report


def compare(config, endpoint_factory, output, **kwargs):
    reports = {}
    initial_assets = task_rows(config, kwargs.get("splits", TEST_SPLITS))[1]
    for profile in ("baseline", "state", "full"):
        if task_rows(config, kwargs.get("splits", TEST_SPLITS))[1] != initial_assets:
            raise ValueError("Dataset changed between comparison profiles")
        reports[profile] = evaluate(config, endpoint_factory(), Path(output) / profile, profile=profile, **kwargs)
    report = {"status": "completed", "shared_config": config, "profiles": {k: v["overall"] for k, v in reports.items()},
              "comparison_scope": "固定同一模型、请求、工具与调用预算；仅控制器状态/校验策略不同；无模型训练"}
    write_json(Path(output) / "comparison.json", report)
    return report


def replay(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshot = result.get("snapshot", result)
    if snapshot["backend"]["kind"] == "http": raise ValueError("Remote writes are never replayed; inspect saved gateway receipts instead")
    backend = restore_backend(snapshot["backend"])
    if backend.kind != "dataset": return {"replayed": True, "offline_validation": None, "model_called": False}
    validation = backend.offline_evaluate()
    if "validation" in result and validation != result["validation"]: raise ValueError("Replay validation differs")
    if result.get("raw_final_validation") is not None and backend.offline_evaluate(snapshot["raw_final"]) != result["raw_final_validation"]:
        raise ValueError("Replay proposed-final validation differs")
    return {"replayed": True, "validation": validation, "model_called": False}
