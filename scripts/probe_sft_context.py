"""Bounded real-trajectory forward/backward probes; no epoch or adapter is saved."""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_cases(dataset, limits):
    from lightagent.sft_data import file_hash, model_fingerprint, read_rows
    root = Path(dataset)
    report = json.loads((root / "preparation-report.json").read_text(encoding="utf-8"))
    for relative in ("train.jsonl", "metadata/train.jsonl"):
        if file_hash(root / relative) != report["asset_sha256"][relative]:
            raise ValueError("Dataset changed: " + relative)
    metadata = list(read_rows(root / "metadata/train.jsonl"))
    selected = []
    for limit in limits:
        eligible = [m for m in metadata if m["token_length"] <= limit]
        meta = max(eligible, key=lambda m: (m["token_length"], m["source"] == "self", m["fingerprint"]))
        selected.append({"target_limit": limit, "meta": meta})
    wanted = {case["meta"]["row_index"] for case in selected}
    records = {n: r for n, r in enumerate(read_rows(root / "train.jsonl")) if n in wanted}
    for case in selected:
        case["record"] = records[case["meta"]["row_index"]]
        if model_fingerprint(case["record"]) != case["meta"]["fingerprint"]:
            raise ValueError("Selected record fingerprint mismatch")
    return selected


def child(args):
    import psutil
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from lightagent.training import encode_record, load_model_and_lora, load_tokenizer, parameter_fingerprint, trainer_row
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    result = {"status": "running", "phase": "initializing", "mode": args.mode,
              "target_limit": case["target_limit"], "actual_tokens": case["meta"]["token_length"],
              "source": case["meta"]["source"], "source_id": case["meta"]["source_id"],
              "fingerprint": case["meta"]["fingerprint"], "optimizer_step_attempted": False,
              "optimizer_step_completed": False, "repeats_completed": 0,
              "saved_adapter": False, "full_training_started": False}
    start = time.monotonic()
    def stage(name, **values):
        result.update(phase=name, elapsed_seconds=round(time.monotonic()-start, 3), **values)
        write(args.result, result)
        print(json.dumps({"phase": name, "mode": args.mode, "tokens": result["actual_tokens"], **values}), flush=True)
    try:
        config = json.loads(Path("configs/sft-v1-p1000.json").read_text(encoding="utf-8"))
        config["data_max_length"] = 4096
        trainer_args = copy.deepcopy(config["trainer"])
        trainer_args.update(output_dir=str(Path(args.result).parent / "trainer-temporary"),
                            gradient_accumulation_steps=1, max_steps=1,
                            eval_strategy="no", save_strategy="no", disable_tqdm=True,
                            activation_offloading=args.mode == "offload", prediction_loss_only=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        tokenizer, template = load_tokenizer(config)
        encoded, inspection = encode_record(tokenizer, case["record"], template, 4096)
        if len(encoded["input_ids"]) != result["actual_tokens"]:
            raise ValueError("Selected trajectory length drift")
        stage("loading_quantized_model", supervised_tokens=sum(encoded["assistant_masks"]))
        model, lora, targets = load_model_and_lora(config)
        row = trainer_row(encoded, case["record"])
        trainer = SFTTrainer(model=model, args=SFTConfig(**trainer_args), processing_class=tokenizer,
                             train_dataset=Dataset.from_list([row]), peft_config=lora)
        trainer.current_gradient_accumulation_steps = 1
        trainer.create_optimizer()
        trainable = [p for p in trainer.model.parameters() if p.requires_grad]
        names = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n or "language_model" not in n for n in names):
            raise ValueError("Probe trainable parameters exceed language LoRA")
        # Allocate real Adam moments before the passes, without updating parameters.
        # This includes optimizer-state memory even in probes that do not take a step.
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                trainer.optimizer.state[p].update(step=torch.tensor(0.0),
                    exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
        torch.cuda.synchronize()
        result.update(load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      model_allocated_bytes=torch.cuda.memory_allocated(), lora_target_modules=len(targets),
                      trainable_parameters=sum(p.numel() for p in trainable),
                      activation_offloading=trainer_args["activation_offloading"])
        batch = trainer.data_collator([row])
        torch.cuda.reset_peak_memory_stats()
        passes = []
        for repeat in range(args.repeats):
            trainer.optimizer.zero_grad(set_to_none=True)
            stage("forward_backward", repeat=repeat+1)
            t0 = time.monotonic()
            loss = trainer.training_step(trainer.model, {k: v.clone() for k, v in batch.items()})
            torch.cuda.synchronize()
            if not math.isfinite(float(loss.item())):
                raise ValueError("Nonfinite training loss")
            grads = [p.grad for p in trainable if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise ValueError("Missing or nonfinite LoRA gradients")
            if not any(torch.count_nonzero(g).item() for g in grads):
                raise ValueError("All LoRA gradients are zero")
            passes.append({"loss": float(loss.item()), "seconds": time.monotonic()-t0})
            result["repeats_completed"] += 1
            stage("pass_complete", passes=passes)
        if args.optimizer_step:
            before = parameter_fingerprint(trainer.model)
            stage("optimizer_step", optimizer_step_attempted=True)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            trainer.optimizer.step()
            torch.cuda.synchronize()
            changed = before != parameter_fingerprint(trainer.model)
            if not changed:
                raise ValueError("Temporary LoRA update did not change parameters")
            result.update(optimizer_step_completed=True, temporary_lora_parameters_changed=True)
        result["status"] = "passed"
        stage("completed")
    except BaseException as exc:
        message = str(exc)
        result.update(status="oom" if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in message.lower() else "error",
                      error_type=type(exc).__name__, error=message[:1600])
        traceback.print_exc()
    finally:
        result.update(elapsed_seconds=time.monotonic()-start, host_rss_bytes=psutil.Process().memory_info().rss)
        if torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          end_allocated_bytes=torch.cuda.memory_allocated())
        write(args.result, result)
    raise SystemExit(0 if result["status"] == "passed" else 2)


def terminate_tree(process):
    import psutil
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child_process in reversed(children):
        try:
            child_process.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


def parent(args):
    import psutil
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("Choose a fresh probe output directory")
    for process in psutil.process_iter(["pid", "cmdline"]):
        line = " ".join(process.info["cmdline"] or [])
        if "lightagent.training" in line and "--execute" in line:
            raise RuntimeError("An existing SFT process is running; not launching a second GPU job")
    root.mkdir(parents=True)
    cases = select_cases(args.dataset, args.limits)
    for case in cases:
        write(root / f"case-{case['target_limit']}.json", case)
    report = {"status": "running", "device_scope": "P1000 NF4 FP32, rank4 q/v, microbatch1, gradient checkpointing",
              "data_scope": "Real complete trajectories from the full train split; no truncation or protocol removal",
              "cases": [{k: v for k, v in c.items() if k != "record"} for c in cases], "trials": [],
              "max_seconds_per_trial": args.max_seconds, "minimum_host_available_GiB": args.min_free_gib,
              "existing_adapters_modified": False, "full_training_started": False}
    path = root / "summary.json"
    write(path, report)
    def run(case, mode, update):
        trial_root = root / f"{mode}-{case['target_limit']}"
        trial_root.mkdir()
        result_path = trial_root / "result.json"
        command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--child",
                   "--case", str(root / f"case-{case['target_limit']}.json"), "--result", str(result_path),
                   "--mode", mode, "--repeats", str(args.repeats)]
        if update:
            command.append("--optimizer-step")
        started = time.monotonic()
        guard, peak_rss, min_available = None, 0, psutil.virtual_memory().available
        print(json.dumps({"trial": mode, "target_limit": case["target_limit"], "actual_tokens": case["meta"]["token_length"]}), flush=True)
        with (trial_root / "process.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    available = psutil.virtual_memory().available
                    min_available = min(min_available, available)
                    try:
                        tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                        peak_rss = max(peak_rss, sum(p.memory_info().rss for p in tree if p.is_running()))
                    except psutil.Error:
                        pass
                    if available < args.min_free_gib * 1024**3:
                        guard = "host_ram_guard"
                    elif time.monotonic()-started > args.max_seconds:
                        guard = "timeout"
                    if guard:
                        terminate_tree(process)
                        break
                    time.sleep(2)
            finally:
                if process.poll() is None:
                    terminate_tree(process)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"status": "process_error"}
        if guard:
            result["status"] = guard
        result.update(mode=mode, target_limit=case["target_limit"], actual_tokens=case["meta"]["token_length"],
                      peak_process_tree_rss_bytes=peak_rss, minimum_host_available_bytes=min_available,
                      wall_seconds=time.monotonic()-started, exit_code=process.returncode,
                      result_file=str(result_path), log_file=str(trial_root / "process.log"))
        write(result_path, result)
        report["trials"].append(result)
        write(path, report)
        print(json.dumps({k: result.get(k) for k in ("mode", "actual_tokens", "status", "phase", "peak_allocated_bytes", "elapsed_seconds", "error")}), flush=True)
        time.sleep(3)
        return result
    for case in cases:
        run(case, "baseline", case is cases[-1])
    last = report["trials"][-1]
    if last["status"] != "passed":
        run(cases[-1], "offload", not any(t.get("optimizer_step_attempted") for t in report["trials"]))
    report["status"] = "completed_probe"
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["optimizer_update_attempts"] = sum(bool(t.get("optimizer_step_attempted")) for t in report["trials"])
    report["optimizer_updates_completed"] = sum(bool(t.get("optimizer_step_completed")) for t in report["trials"])
    write(path, report)
    write("reports/p1000-long-context-report.json", report)
    print(json.dumps({"probe_complete": True, "report": "reports/p1000-long-context-report.json"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--result")
    parser.add_argument("--mode", choices=["baseline", "offload"], default="baseline")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dataset", default="data/sft_v1")
    parser.add_argument("--output", default="artifacts/p1000-long-context-probe-v1")
    parser.add_argument("--limits", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--max-seconds", type=int, default=360)
    parser.add_argument("--min-free-gib", type=float, default=1.0)
    args = parser.parse_args()
    if args.child:
        child(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
