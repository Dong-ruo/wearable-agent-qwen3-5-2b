from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .evaluation import TEST_SPLITS, compare, evaluate, load_config, replay, task_rows
from .interactive_evaluation import VARIANTS, evaluate_interactive, load_cases
from .protocol import ChatEndpoint
from .service import SessionManager, serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定模型八工具助手；不训练、不下载模型；网络推理需 --execute")
    parser.add_argument("--config", default="configs/assistant.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    for name in ("evaluate", "compare"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
        cmd.add_argument("--limit", type=int)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--resume", action="store_true")
        cmd.add_argument("--execute", action="store_true")
        if name == "evaluate": cmd.add_argument("--profile", choices=["baseline", "state", "full"], default="full")
    cmd = commands.add_parser("replay")
    cmd.add_argument("--artifact", required=True)
    cmd = commands.add_parser("interactive")
    cmd.add_argument("--cases-file", default="examples/assistant-interactive-cases.json")
    cmd.add_argument("--case", action="append", dest="case_ids")
    cmd.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    cmd.add_argument("--max-model-calls", type=int)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--execute", action="store_true")
    for name in ("chat", "serve"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--database", default="outputs/assistant/sessions.sqlite3" if name == "serve" else "outputs/assistant/chat.sqlite3")
        cmd.add_argument("--execute", action="store_true")
        if name == "serve":
            cmd.add_argument("--host", default="127.0.0.1")
            cmd.add_argument("--port", type=int, default=8765)
        else:
            cmd.add_argument("--session")
            cmd.add_argument("--profile", choices=["baseline", "state", "full"], default="full")
    args = parser.parse_args(argv)
    if args.command == "replay":
        print(json.dumps(replay(args.artifact), ensure_ascii=False, indent=2)); return
    config = load_config(args.config)
    if args.command == "interactive":
        cases = load_cases(args.cases_file, args.case_ids)
        if args.max_model_calls is not None:
            if args.max_model_calls < 1: parser.error("max-model-calls must be positive")
            config["runtime"]["max_model_calls"] = args.max_model_calls
        if not args.execute:
            print(json.dumps({"executed": False, "cases": [row["id"] for row in cases],
                              "variants": args.variants, "runtime_budget": config["runtime"]}, ensure_ascii=False, indent=2))
            return
        report = evaluate_interactive(config, lambda: ChatEndpoint(config["endpoint"]), args.cases_file, args.output,
                                      args.variants, args.case_ids)
        print(json.dumps({name: variant["summary"] for name, variant in report["variants"].items()}, ensure_ascii=False, indent=2))
        return
    if args.command == "plan" or not args.execute:
        splits = getattr(args, "splits", TEST_SPLITS)
        rows, _ = task_rows(config, splits)
        print(json.dumps({"executed": False, "training": False, "model": config["endpoint"]["model"],
            "endpoint": config["endpoint"]["base_url"], "tasks_available": len(rows), "profiles": ["baseline", "state", "full"],
            "runtime_budget": config["runtime"], "note": "未调用模型；当前所有服务与工具连接均未验证，正式运行需显式 --execute"}, ensure_ascii=False, indent=2)); return
    endpoint = ChatEndpoint(config["endpoint"])
    if args.command in {"evaluate", "compare"}:
        if args.limit is not None and args.limit < 1: parser.error("limit must be positive")
        kwargs = dict(output=args.output, splits=args.splits, limit=args.limit, resume=args.resume)
        result = evaluate(config, endpoint, profile=args.profile, **kwargs) if args.command == "evaluate" else compare(config, lambda: ChatEndpoint(config["endpoint"]), **kwargs)
        print(json.dumps({k: v for k, v in result.items() if k in {"status", "overall", "profiles"}}, ensure_ascii=False, indent=2))
    elif args.command == "serve":
        serve(config, endpoint, args.host, args.port, args.database)
    else:
        manager = SessionManager(config, endpoint, args.database)
        session_id = args.session
        try:
            while True:
                text = input("你（/quit退出，/cancel停止）> ").strip()
                if text == "/quit": break
                if not text: continue
                result = manager.message(session_id, text, uuid.uuid4().hex) if session_id else manager.create(text, uuid.uuid4().hex, args.profile)
                session_id = result["session_id"]
                print(json.dumps({"session_id": session_id, "state": result["state"], "final": result["final"],
                     "latest": result["messages"][-1]}, ensure_ascii=False, indent=2))
        finally: manager.store.close()


if __name__ == "__main__": main()
