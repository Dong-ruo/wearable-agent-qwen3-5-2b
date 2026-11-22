from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="CPU 标准库穿戴助手任务工厂；不会下载模型或调用外部 API")
    commands = parser.add_subparsers(dest="command", required=True)
    checks = commands.add_parser("check", help="按 core → L1-L3 → L4-L6 → pipeline 顺序运行 unittest")
    checks.add_argument("--output", default="outputs/wearable_tasks/unit-tests.json")
    build_cmd = commands.add_parser("build", help="生成、执行参考策略、批量验证、导出，并写入实测报告")
    build_cmd.add_argument("--config", required=True)
    build_cmd.add_argument("--output", required=True)
    build_cmd.add_argument("--checks-report")
    build_cmd.add_argument("--smoke-report")
    build_cmd.add_argument("--seed", type=int)
    build_cmd.add_argument("--overwrite", action="store_true")
    replay_cmd = commands.add_parser("replay", help="从私有库加载指定任务，并重放其已保存轨迹")
    replay_cmd.add_argument("--directory", required=True)
    replay_cmd.add_argument("--case-id", required=True)
    commands.add_parser("schema", help="只打印八个工具的冻结 JSON Schema")
    args = parser.parse_args(argv)
    if args.command == "check":
        from .checks import run_checks
        result = run_checks(Path(args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "passed" else 1
    if args.command == "schema":
        from .contract import schema_hash, tool_schemas
        print(json.dumps({"hash": schema_hash(), "tools": tool_schemas()}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build":
        from .pipeline import build, load_configuration
        config = load_configuration(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        report = build(config, Path(args.output), args.checks_report, args.smoke_report, args.overwrite)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    from .exporting import TaskStore
    from .reference import replay
    store = TaskStore(args.directory)
    task = store.get(args.case_id)
    path = Path(args.directory) / "private/trajectories" / (task["split"]+".jsonl")
    trajectory = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value["case_id"] == args.case_id:
                trajectory = value
                break
    if trajectory is None:
        raise ValueError("该任务的轨迹不存在")
    env = replay(task, trajectory)
    print(json.dumps(env.evaluate(), ensure_ascii=False, indent=2))
    return 0 if env.evaluate()["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
