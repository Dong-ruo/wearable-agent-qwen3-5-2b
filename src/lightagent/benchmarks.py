"""Official EvalScope BFCL launcher. No benchmark downloads or calls without --execute."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import sys
from pathlib import Path


SUBSETS = {"simple", "multiple", "parallel", "irrelevance", "multi_turn_base", "multi_turn_miss_param", "multi_turn_miss_func"}


def benchmark_plan(config, formal=False):
    config = copy.deepcopy(config)
    subsets = config["dataset_args"]["bfcl_v3"]["subset_list"]
    if config["datasets"] != ["bfcl_v3"] or not subsets or not set(subsets) <= SUBSETS or len(subsets) != len(set(subsets)):
        raise ValueError("BFCL 类别配置不合法")
    if formal:
        config.pop("limit", None)
        config["work_dir"] = "outputs/bfcl-fixed-subsets"
    if not formal and not config.get("limit"):
        raise ValueError("冒烟评测必须显式限制样本数；正式子集请用 --formal")
    return {"executed": False, "scope": "BFCL-v3 固定七类子集（非 BFCL 总分）" if formal else "BFCL-v3 接口冒烟（非正式成绩）",
            "subsets": subsets, "sample_counts": None, "config": config,
            "note": "样本数和指标必须从实际 EvalScope 报告提取，不使用计划数冒充完成数"}


def main():
    from .console import utf8_console
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bfcl-smoke.json")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = benchmark_plan(json.loads(Path(args.config).read_text(encoding="utf-8-sig")), args.formal)
    if args.execute:
        if sys.version_info < (3, 11) or importlib.metadata.version("evalscope") != "1.15.0":
            raise RuntimeError("请使用独立 Python >=3.11、evalscope[bfcl]==1.15.0 环境")
        from evalscope import run_task
        from evalscope.config import TaskConfig
        config = {**report["config"], "api_key": os.environ.get("MODEL_API_KEY", "EMPTY")}
        run_task(task_cfg=TaskConfig(**config))
        report["executed"] = True
        report["note"] = "已完成 EvalScope 调用，实际样本数和分数见 work_dir 的原始报告"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
