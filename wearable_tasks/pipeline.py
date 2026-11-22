"""Single-process generation, reference execution, adversarial checks and export orchestration."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

from . import GENERATOR_VERSION, SCHEMA_VERSION, TOOL_VERSION
from .auditing import audit_composition_execution, audit_isolation, audit_negatives, audit_privacy, audit_splits
from .contract import canonical, digest, schema_hash, tool_schemas
from .exporting import grpo_reference, sft_record
from .reference import replay, run_reference
from .templates import build_case, plan_groups


def source_fingerprint():
    hasher = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def load_configuration(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config["profile"] not in {"smoke", "formal"} or type(config["seed"]) is not int:
        raise ValueError("未知配置或无效种子")
    target = 100 if config["profile"] == "smoke" else 2900
    if sum(config["counts"].values()) != target:
        raise ValueError("配置规模不符")
    plan_groups(config["counts"], config["seed"])
    return config


def build_tasks(config):
    groups = plan_groups(config["counts"], config["seed"])
    return [build_case(group, variant) for group in groups for variant in range(4)]


def dataset_hash(tasks):
    hasher = hashlib.sha256()
    for task in tasks:
        hasher.update((canonical(task)+"\n").encode())
    return hasher.hexdigest()


def check_prerequisites(config, checks_report=None, smoke_report=None):
    fingerprint = source_fingerprint()
    if checks_report:
        checks = json.loads(Path(checks_report).read_text(encoding="utf-8"))
        if checks.get("status") != "passed" or checks.get("source_fingerprint") != fingerprint:
            raise ValueError("单元测试未通过或代码已变化，需要重新运行检查")
    if config["profile"] == "formal":
        if not checks_report or not smoke_report:
            raise ValueError("正式生成必须先提供通过的单元测试和 100 条冒烟报告")
        report = json.loads(Path(smoke_report).read_text(encoding="utf-8"))
        if report.get("status") != "passed" or report.get("profile") != "smoke" or report.get("reference_successes") != 100 or report.get("task_count") != 100:
            raise ValueError("100 条冒烟任务尚未全部验证成功")
        if report.get("source_fingerprint") != fingerprint or report.get("schema_hash") != schema_hash() or report.get("seed") != config["seed"]:
            raise ValueError("冒烟报告的代码、契约或种子与正式配置不一致")


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build(config, output, checks_report=None, smoke_report=None, overwrite=False):
    check_prerequisites(config, checks_report, smoke_report)
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError("输出目录非空；请使用新目录，或明确传 --overwrite（只覆盖本模块的已知文件）")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tasks = build_tasks(config)
    split_checks = audit_splits(tasks)
    expected_hash = dataset_hash(tasks)
    if dataset_hash(build_tasks(config)) != expected_hash:
        raise ValueError("同种子重建结果不一致")
    report = {"status": "running", "python_version": platform.python_version(), "seed": config["seed"],
        "schema_version": SCHEMA_VERSION, "tool_version": TOOL_VERSION, "generator_version": GENERATOR_VERSION,
        "schema_hash": schema_hash(), "source_fingerprint": source_fingerprint(), "dataset_sha256": expected_hash,
        "profile": config["profile"], "task_count": len(tasks), "reference_successes": 0,
        "counts": {"split": dict(Counter(t["split"] for t in tasks)), "level": dict(Counter(str(t["level"]) for t in tasks)),
                   "template": dict(Counter(t["template_id"] for t in tasks)), "branch": dict(Counter(t["branch"] for t in tasks))},
        "checks": {"splits": split_checks, "fixed_seed_rebuild": True}, "failures": [],
        "memory": {"status": "未测量", "bytes": None},
        "limitations": ["参考通过率验证可解性与环境一致性，不是模型成功率。", "结构化 ask_user 未验证自由文本澄清识别。",
            "SFT/GRPO 导出是框架无关格式，尚未接入 TRL 原生 Trainer 与真实 tokenizer loss mask。",
            "不进行真实通知、外部 API、模型调用或健康建议。", "工具集合仅八个业务工具，首版固定 +08:00。"]}
    handles = {}
    staged_paths = []
    calls, negative_counts, summaries = [], Counter(), {}
    isolation_count = replay_count = privacy_count = 0
    negative_examples = {}
    try:
        for split in config["counts"]:
            for kind in ("private", "private/trajectories", "public/grpo"):
                path = output / kind / (split+".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix+".part")
                handles[(kind, split)] = temporary.open("w", encoding="utf-8", newline="\n")
                staged_paths.append((temporary, path))
            if split in {"train", "val"}:
                path = output / "public/sft" / (split+".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix+".part")
                handles[("public/sft", split)] = temporary.open("w", encoding="utf-8", newline="\n")
                staged_paths.append((temporary, path))
        execution_started = time.perf_counter()
        # Gate later levels on completion of every L1-L3 task first.
        for phase, levels in (("L1-L3", {1, 2, 3}), ("L4-L6", {4, 5, 6})):
            for task in tasks:
                if task["level"] not in levels:
                    continue
                try:
                    env = run_reference(task)
                    validation = env.evaluate()
                    if not validation["success"]:
                        raise ValueError("参考执行失败: " + ",".join(validation["failure_codes"]))
                    trajectory = env.trajectory()
                    replay(task, trajectory)
                    replay_count += 1
                    audit_isolation(task)
                    isolation_count += 1
                    audit_privacy(task, env)
                    privacy_count += 1
                    counts, examples = audit_negatives(task, env)
                    negative_counts.update(counts)
                    for example in examples:
                        negative_examples.setdefault(example["type"], example)
                    split = task["split"]
                    rows = [("private", task), ("private/trajectories", trajectory), ("public/grpo", grpo_reference(task))]
                    if split in {"train", "val"}:
                        rows.append(("public/sft", sft_record(task, trajectory, verified_env=env)))
                    for kind, row in rows:
                        handles[(kind, split)].write(canonical(row)+"\n")
                    calls.append(env._calls)
                    summaries[task["case_id"]] = {"names": [r["name"] for r in env._log if r["result"]["status"] == "success"], "followup_fired": env._followup_fired}
                    report["reference_successes"] += 1
                except Exception as exc:
                    report["failures"].append({"case_id": task["case_id"], "phase": phase, "error": type(exc).__name__, "reason": str(exc)})
                    raise
            report["checks"][phase] = {"passed": True, "validated": sum(t["level"] in levels for t in tasks)}
        report["reference_and_validation_seconds"] = time.perf_counter()-execution_started
        report["checks"]["composition"] = audit_composition_execution(tasks, [summaries[t["case_id"]] for t in tasks])
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["pipeline_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        for handle in handles.values():
            handle.close()
        report["file_sha256"] = {}
        if report["status"] == "passed":
            for temporary, path in staged_paths:
                temporary.replace(path)
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024*1024), b""):
                        hasher.update(chunk)
                report["file_sha256"][path.relative_to(output).as_posix()] = hasher.hexdigest()
        report["elapsed_seconds"] = time.perf_counter()-started
        report["negative_samples_rejected"] = dict(negative_counts)
        report["negative_samples_rejected_total"] = sum(negative_counts.values())
        report["checks"].update(isolation_cases=isolation_count, replay_cases=replay_count, privacy_cases=privacy_count)
        report["tool_calls"] = {"total": sum(calls), "histogram": dict(Counter(str(n) for n in calls)),
            "mean": statistics.mean(calls) if calls else None, "median": statistics.median(calls) if calls else None,
            "p95": sorted(calls)[max(0, math.ceil(.95*len(calls))-1)] if calls else None, "max": max(calls) if calls else None}
        _write_report(output / "validation-report.json", report)
        _write_report(output / "private/negative-examples.json", list(negative_examples.values()))
        _write_report(output / "public/tool-schema.json", {"tool_version": TOOL_VERSION, "schema_hash": schema_hash(), "tools": tool_schemas()})
        _write_report(output / "private/group-plan.json", plan_groups(config["counts"], config["seed"]))
    if report["status"] != "passed":
        raise ValueError("任务生成或验证未通过，详见 " + str(output / "validation-report.json"))
    return report
