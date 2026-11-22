"""Re-execute an explicit artifact and independently recompute its reward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environment import Environment
from .goals import EpisodeSpec
from .models import Final
from .rewards import score_episode
from .user_simulator import DeterministicUser


def replay(spec: EpisodeSpec, artifact: dict):
    env = Environment(spec.scenario)
    env._user = DeterministicUser(spec.missing_slots, spec.user_answers)
    for row in artifact["trace"]:
        observation = env.execute(row["name"], row["arguments"])
        if observation != row["result"]:
            raise ValueError(f"第 {row['index']} 次工具观察与重放不一致")
    if env.snapshot() != artifact["state"]:
        raise ValueError("轨迹重放的终态与保存终态不一致")
    if "raw_finals" not in artifact:
        raise ValueError("缺少策略原始反馈，不能用被 Runtime 修正后的结果计算奖励")
    final = Final.model_validate(artifact["raw_finals"][-1]) if artifact["raw_finals"] else None
    return score_episode(spec.goal, env, final)


def main():
    from .console import utf8_console
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    spec = EpisodeSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8-sig"))
    report = replay(spec, json.loads(Path(args.artifact).read_text(encoding="utf-8-sig")))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
