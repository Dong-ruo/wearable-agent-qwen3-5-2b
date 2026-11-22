"""Run an explicitly supplied private-goal task. Does not generate or collect datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import dump
from .environment import Environment
from .goals import EpisodeSpec
from .models import Final
from .policy import EndpointPolicy, RulePolicy
from .rewards import score_episode
from .runtime import Runtime
from .user_simulator import DeterministicUser


def run_episode(spec, policy, max_steps=20, max_user_turns=5):
    env = Environment(spec.scenario)
    env._user = DeterministicUser(spec.missing_slots, spec.user_answers)
    runtime = Runtime(env, policy, max_steps=max_steps, goal=spec.goal)
    runtime.run(spec.request)
    for _ in range(max_user_turns):
        answer = env._user.respond(runtime.last_final)
        if answer is None:
            break
        runtime.run(answer)
    raw = Final.model_validate(runtime.raw_finals[-1]) if runtime.raw_finals else None
    report = score_episode(spec.goal, env, raw)
    return runtime, report


def main():
    from .console import utf8_console
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--backend", choices=["rule", "openai"], default="openai")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    spec = EpisodeSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8-sig"))
    policy = RulePolicy() if args.backend == "rule" else EndpointPolicy(args.base_url, args.model)
    runtime, report = run_episode(spec, policy, args.max_steps)
    dump(args.output, {**runtime.artifact(), "reward_report": report})
    print(json.dumps({"final": runtime.last_final.model_dump(), "reward": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
