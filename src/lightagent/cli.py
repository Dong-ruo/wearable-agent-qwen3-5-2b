from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .cases import REQUEST, cases
from .environment import Environment, tool_specs
from .evaluation import aggregate, assess
from .models import Scenario
from .policy import EndpointPolicy, RulePolicy
from .runtime import Runtime
from .user_simulator import DeterministicUser


def dump(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_policy(args):
    if args.backend == "rule":
        return RulePolicy()
    return EndpointPolicy(args.base_url, args.model, args.label, timeout=args.timeout)


def make_runtime(args, scenario, env=None):
    runtime = Runtime(env if env is not None else Environment(scenario), make_policy(args), args.max_steps)
    if args.trace:
        def show(row):
            status = "OK" if row["result"]["ok"] else row["result"]["error"]["code"]
            print(f"  [{row['index'] + 1:02}] {row['name']} {json.dumps(row['arguments'], ensure_ascii=False)} -> {status}")
        runtime.on_tool = show
    return runtime


def show_final(final):
    print(f"[{final.status}] {final.message}")


def parser():
    root = argparse.ArgumentParser(description="本地轻量任务助手（当前阶段不训练）")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("demo", "chat", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=["rule", "openai"], default="rule", help="openai 表示兼容协议端点，不限定供应商")
        p.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
        p.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-2B"))
        p.add_argument("--label", default="base", help="评测标签，例如 base/sft/rl；不会启动训练")
        p.add_argument("--timeout", type=int, default=60)
        p.add_argument("--max-steps", type=int, default=20)
        p.add_argument("--trace", action="store_true")
        p.add_argument("--output", help="将执行轨迹或评测结果保存为 JSON")
        if name == "demo":
            p.add_argument("--case", choices=list(cases()), default="sunny")
            p.add_argument("--reply", action="append", default=[], help="提供澄清回复，可重复")
        elif name == "chat":
            p.add_argument("--scenario", help="自定义模拟环境 JSON；不指定则使用当前时间及模拟晴天天气")
        else:
            p.add_argument("--case", choices=list(cases()), action="append", help="不指定则执行全部固定回归场景")
    sub.add_parser("cases", help="列出固定回归场景")
    sub.add_parser("tools", help="输出工具 JSON Schema")
    return root


def main(argv=None):
    # Windows defaults can otherwise corrupt the Chinese demo and redirected JSON.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    if args.command == "cases":
        for case in cases().values():
            print(f"{case.name:26} {case.description}")
        return 0
    if args.command == "tools":
        print(json.dumps(tool_specs(), ensure_ascii=False, indent=2))
        return 0
    if args.max_steps < 1 or args.timeout < 1:
        raise SystemExit("max-steps 和 timeout 必须大于 0")
    if args.command == "demo":
        case = cases()[args.case]
        runtime = make_runtime(args, case.scenario)
        print(f"后端：{runtime.policy.label}；环境：{case.description}；模拟当前时间：{runtime.env.now.isoformat()}")
        print("用户：" + case.request)
        show_final(runtime.run(case.request))
        for reply in (args.reply or case.replies):
            if runtime.last_final.status != "clarification":
                break
            print("用户：" + reply)
            show_final(runtime.run(reply))
        if args.output:
            dump(args.output, runtime.artifact())
        # A demo may intentionally end in failure or use a custom clarification reply.
        # Only the eval command uses expected outcomes to determine its exit status.
        return 0
    if args.command == "chat":
        if args.scenario:
            scene = Scenario.model_validate_json(Path(args.scenario).read_text(encoding="utf-8-sig"))
        else:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            weather = {f"{city}|{(now + timedelta(days=i)).date().isoformat()}": "sunny"
                       for city in ("杭州", "北京", "上海", "深圳", "成都", "南京", "广州", "苏州") for i in range(4)}
            scene = Scenario(name="interactive-mock", now=now, weather=weather)
        runtime = make_runtime(args, scene)
        print(f"后端：{runtime.policy.label}；当前时间：{runtime.env.now.isoformat()}。天气、日程与提醒全部为本地模拟，不会推送通知。")
        print("输入 /quit 退出，/state 查看状态，/new 开始新任务（保留日程和提醒）。")
        try:
            while True:
                text = input("你 > ").strip()
                if text == "/quit":
                    break
                if text == "/state":
                    print(json.dumps(runtime.env.snapshot(), ensure_ascii=False, indent=2))
                    continue
                if text == "/new":
                    previous = runtime
                    runtime = make_runtime(args, scene, env=previous.env)
                    print("已开始新任务，已有日程和提醒已保留。")
                    continue
                if not text:
                    continue
                if runtime.last_final and runtime.last_final.status != "clarification":
                    print("任务已结束，输入 /new 开始新任务，或 /quit 退出。")
                    continue
                show_final(runtime.run(text))
                if args.output:
                    dump(args.output, runtime.artifact())
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
        return 0
    rows, artifacts = [], []
    for name in args.case or list(cases()):
        case = cases()[name]
        runtime = make_runtime(args, case.scenario)
        runtime.run(case.request)
        if case.replies:
            slot = "city" if "city" in name else "expression" if "time_followup" == name else "confirm_time"
            simulated_user = DeterministicUser({slot}, {slot: case.replies[0]})
            reply = simulated_user.respond(runtime.last_final)
            if reply is not None:
                runtime.run(reply)
        row = assess(case, runtime)
        rows.append(row)
        artifacts.append(runtime.artifact())
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name:26} {row['status']:14} tools={row['tool_calls']} {' '.join(row['failures'])}")
    report = aggregate(rows, runtime.policy.label)
    print(f"结果正确率 {report['outcome_accuracy']:.1%}；完整完成率 {report['task_completion_rate']:.1%}；平均工具调用 {report['mean_tool_calls']:.1f}")
    print(report["note"])
    if args.output:
        dump(args.output, {**report, "artifacts": artifacts})
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
