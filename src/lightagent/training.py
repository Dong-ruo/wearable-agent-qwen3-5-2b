"""TRL training entrypoints. Default is a dependency-free plan; --execute is explicit."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .data_pipeline import read_rows, validate_record
from .goals import EpisodeSpec

TRL_VERSION = "1.18.0"


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("framework") != "trl" or config.get("trl_version") != TRL_VERSION:
        raise ValueError(f"只接受已核查的 TRL {TRL_VERSION} 配置")
    if config["stage"] not in {"sft", "grpo"}:
        raise ValueError("未知训练阶段")
    if config["stage"] == "sft" and (not config["trainer"].get("assistant_only_loss") or config["trainer"].get("packing")):
        raise ValueError("SFT 必须启用 assistant_only_loss 并关闭 packing 以审计 mask")
    hardware = config.get("hardware", {})
    if hardware.get("load_in_4bit") and (config["stage"] != "sft" or hardware.get("device") != "cuda" or config.get("export_merged")):
        raise ValueError("4-bit 配置仅支持 CUDA SFT，且必须关闭在线合并导出")
    if hardware.get("dtype", "bfloat16") not in {"float32", "float16", "bfloat16"}:
        raise ValueError("不支持的模型精度")
    if hardware.get("device") == "cpu" and not config["trainer"].get("use_cpu"):
        raise ValueError("CPU 配置必须设置 use_cpu=true")
    if config["stage"] == "grpo":
        args = config["trainer"]
        if args["num_generations"] < 2 or args["generation_batch_size"] % args["num_generations"]:
            raise ValueError("GRPO group >=2，generation_batch_size 必须能被 group 整除")
        if config["per_turn_max_new_tokens"] > args["max_completion_length"]:
            raise ValueError("单轮输出预算不得超过整条轨迹预算")
        if args.get("generation_kwargs"):
            raise ValueError("本版本不允许额外 generation_kwargs 覆盖已审计的输出预算")
    return config


def plan(config):
    paths = [config["model_name_or_path"]]
    paths += [config[k] for k in ("train_file", "validation_file", "episodes_file", "validation_episodes_file") if config.get(k)]
    model_dir = Path(config["model_name_or_path"])
    weight_files = sorted([*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")]) if model_dir.is_dir() else []
    return {"stage": config["stage"], "framework": "trl", "trl_version": TRL_VERSION,
            "executed": False, "requires_training_python": ">=3.11", "python": platform.python_version(),
            "missing_local_paths": [p for p in paths if not Path(p).exists()],
            "local_model_weights_present": bool(weight_files),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "config": config, "note": "仅配置计划，不下载模型、不读取或生成训练数据、不创建 Trainer"}


def assert_mask(input_ids, mask):
    if len(input_ids) != len(mask) or not input_ids or not any(mask) or not all(x in {0, 1} for x in mask):
        raise ValueError("assistant mask 缺失、长度不符或全零")
    return [token if enabled else -100 for token, enabled in zip(input_ids, mask)]


def render_message_prefixes(tokenizer, record, template, rendered):
    """Audit native render boundaries, including Qwen's system and parallel tool block."""
    messages, previous, prefixes = record["messages"], "", []
    qwen = rendered.startswith("<|im_start|>")
    for i, message in enumerate(messages):
        if qwen and i == 0 and message["role"] == "system":
            # Qwen rejects a system-only conversation. Render the first user too and
            # remove its exact native suffix to obtain the genuine system boundary.
            if len(messages) < 2 or messages[1]["role"] != "user":
                raise ValueError("Qwen system 后必须有 user 才能审计模板边界")
            first = tokenizer.apply_chat_template(messages[:2], tools=record["tools"], chat_template=template,
                enable_thinking=False, tokenize=False, add_generation_prompt=False)
            suffix = "<|im_start|>user\n" + messages[1]["content"].strip() + "<|im_end|>\n"
            if not first.endswith(suffix):
                raise ValueError("Qwen 首个 user 边界无法可靠识别")
            prefix = first[:-len(suffix)]
        else:
            prefix = tokenizer.apply_chat_template(messages[:i+1], tools=record["tools"], chat_template=template,
                enable_thinking=False, tokenize=False, add_generation_prompt=False)
            # Qwen combines consecutive tool observations in one user block. A
            # partial rendering closes that block early; remove only that terminator.
            if (qwen and message["role"] == "tool" and i+1 < len(messages)
                    and messages[i+1]["role"] == "tool" and not rendered.startswith(prefix)
                    and prefix.endswith("<|im_end|>\n")):
                prefix = prefix[:-len("<|im_end|>\n")]
        if not prefix.startswith(previous) or not rendered.startswith(prefix):
            raise ValueError("训练模板不保持前缀，多轮工具协议需先修复")
        prefixes.append(prefix)
        previous = prefix
    return prefixes


def encode_record(tokenizer, record, template, max_length):
    validate_record(record)
    encoded = tokenizer.apply_chat_template(record["messages"], tools=record["tools"], chat_template=template,
        enable_thinking=False, tokenize=True, add_generation_prompt=False, return_dict=True,
        return_assistant_tokens_mask=True)
    ids, mask = encoded["input_ids"], encoded.get("assistant_masks", [])
    if len(ids) > max_length:
        raise ValueError("轨迹超长，禁止截断工具协议；请在数据阶段剔除或重构")
    labels = assert_mask(ids, mask)
    # Audit full role spans with character offsets from the same native template.
    rendered = tokenizer.apply_chat_template(record["messages"], tools=record["tools"], chat_template=template,
        enable_thinking=False, tokenize=False, add_generation_prompt=False)
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if offsets["input_ids"] != ids:
        raise ValueError("模板编码与 offset 编码不一致，不能可靠审计 mask")
    previous = ""
    segments = []
    prefixes = render_message_prefixes(tokenizer, record, template, rendered)
    for message, prefix in zip(record["messages"], prefixes):
        indices = [j for j, (lo, hi) in enumerate(offsets["offset_mapping"]) if lo >= len(previous) and hi <= len(prefix) and hi > lo]
        supervised = [j for j in indices if mask[j]]
        if message["role"] != "assistant" and supervised:
            raise ValueError("用户/系统/工具结果出现未屏蔽的 label")
        if message["role"] == "assistant" and not supervised:
            raise ValueError("assistant 工具调用或最终回复没有监督 token")
        segments.append({"role": message["role"], "supervised_tokens": len(supervised),
                         "supervised_text": tokenizer.decode([ids[j] for j in supervised])})
        previous = prefix
    return {"input_ids": ids, "attention_mask": [1]*len(ids), "assistant_masks": mask, "labels": labels}, {"tokens": len(ids), "segments": segments, "labels": labels}


def require_stack():
    if sys.version_info < (3, 11):
        raise RuntimeError("TRL 1.18.0 训练需 Python >=3.11，请使用独立训练环境")
    if importlib.metadata.version("trl") != TRL_VERSION:
        raise RuntimeError(f"需要锁定 trl=={TRL_VERSION}，不可静默使用其他 API")


def load_tokenizer(config):
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template, is_chat_template_stop_token_trained
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], local_files_only=True)
    template = get_training_chat_template(tokenizer, require_prefix_preservation=True)
    if not is_chat_template_stop_token_trained(tokenizer, chat_template=template):
        raise ValueError("assistant 结束 token 未被监督，禁止启动训练")
    if tokenizer.pad_token_id is None:
        raise ValueError("原生 tokenizer 没有 pad token，必须先确认模型配置")
    return tokenizer, template


def load_model_and_lora(config):
    import torch
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM
    model_path = config["model_name_or_path"]
    mc = AutoConfig.from_pretrained(model_path, local_files_only=True)
    loader = AutoModelForMultimodalLM if mc.model_type == "qwen3_5" else AutoModelForCausalLM
    hardware = config.get("hardware", {})
    dtype = getattr(torch, hardware.get("dtype", "bfloat16"))
    kwargs = {"local_files_only": True, "dtype": dtype,
              "attn_implementation": hardware.get("attention", "sdpa")}
    if hardware.get("device") == "cpu":
        kwargs["device_map"] = {"": "cpu"}
    if hardware.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32))
    model = loader.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    if hardware.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and
               "language_model" in name and name.rsplit(".", 1)[-1] in config["lora"]["target_suffixes"]]
    if not targets:
        raise ValueError("未定位语言侧 LoRA 层；禁止退化为训练全部/视觉层")
    lora = LoraConfig(r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"], lora_dropout=config["lora"].get("dropout", 0.05),
                      target_modules=targets, task_type="CAUSAL_LM", bias="none")
    return model, lora, targets


@contextlib.contextmanager
def turn_budget(trainer, limit):
    """Pinned to TRL 1.18.0; keep trajectory accounting untouched while capping a generation."""
    owner, attribute = (trainer.vllm_generation, "max_completion_length") if trainer.use_vllm else (trainer.generation_config, "max_new_tokens")
    old = getattr(owner, attribute)
    setattr(owner, attribute, min(limit, old) if old is not None else limit)
    try:
        yield
    finally:
        setattr(owner, attribute, old)


def make_budgeted_trainer(base_class, limit):
    class BudgetedGRPOTrainer(base_class):
        def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
            with turn_budget(self, limit):
                result = super()._generate_single_turn(prompt_ids, images, multimodal_fields, num_generations, has_tool_images)
            if any(len(ids) > limit for ids in result[0]):
                raise RuntimeError("TRL 单轮预算没有生效，禁止继续训练")
            return result
    return BudgetedGRPOTrainer


def capture_environment():
    packages = {}
    for name in ("trl", "transformers", "torch", "peft", "accelerate", "datasets", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu, gpu_error = None, None
    if shutil.which("nvidia-smi"):
        try:
            # Windows utilities may emit GBK/OEM diagnostics even under Python -X utf8.
            # Capture bytes so subprocess reader threads never attempt a strict decode.
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                 capture_output=True, check=False, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu_error = type(exc).__name__
    def decode(raw):
        try:
            return raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            return raw.decode("mbcs" if sys.platform == "win32" else "utf-8", errors="replace").strip()
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "gpu": decode(gpu.stdout) if gpu else None,
            "gpu_query_stderr": decode(gpu.stderr) if gpu else None,
            "gpu_query_returncode": gpu.returncode if gpu else None, "gpu_query_error": gpu_error}


def parameter_fingerprint(model):
    hasher = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            hasher.update(name.encode())
            hasher.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def trainer_row(encoded, record):
    # Format detection needs a messages column. Keeping arbitrary tool argument
    # objects there makes Arrow unify unrelated APIs' parameter types. These
    # role/content mirrors are auxiliary; the full protocol is already in IDs/labels.
    return {**encoded, "messages": [{"role": m["role"], "content": m["content"]} for m in record["messages"]]}


def prepare_sft(config, tokenizer, template):
    from datasets import Dataset
    sets, audit, split_groups = {}, [], {}
    preparation = {}
    audit_buckets = defaultdict(list)
    for split, path in (("train", config["train_file"]), ("validation", config["validation_file"])):
        rows, groups = [], set()
        skipped_records, sources, behaviors = [], Counter(), Counter()
        skipped = 0
        limit = config.get("sample_limits", {}).get(split)
        unified = config.get("data_format") == "unified-sft-v1"
        if unified:
            from .sft_data import training_items
            inputs = [(i["record"], i["meta"]) for i in training_items(config, split, tokenizer, template)]
        else:
            inputs = [(record, record.get("metadata", {})) for record in read_rows(path)]
        # Check all declared groups, including rows beyond the local sample limit.
        groups.update(meta.get("group_id") for _, meta in inputs)
        for record, meta in inputs:
            if not unified and (meta.get("source") not in {"Salesforce/xlam-function-calling-60k", "Team-ACE/ToolACE"} or meta.get("license") not in {"CC-BY-4.0", "Apache-2.0"} or meta.get("transform_version") != "public-tools-v1" or meta.get("split") != split or not meta.get("group_id")):
                raise ValueError("旧格式要求公开来源及内联元数据；统一 SFT 请使用 data_format=unified-sft-v1 及独立元数据")
            try:
                encoded, inspection = encode_record(tokenizer, record, template, config["data_max_length"])
            except ValueError as exc:
                if config.get("skip_overlength", False) and str(exc).startswith("轨迹超长"):
                    skipped += 1
                    skipped_records.append({"fingerprint": meta.get("fingerprint"), "source_id": meta.get("source_id"),
                                            "token_length": meta.get("token_length"), "reason": "complete_record_exceeds_length_limit"})
                    continue
                raise
            # TRL checks conversational format even for pre-encoded datasets. Keep
            # public messages for that check; the collator forwards only IDs/labels.
            rows.append(trainer_row(encoded, record))
            sources[meta.get("source", "unknown")] += 1
            behaviors[meta.get("primary_behavior", "unknown")] += 1
            groups.add(meta["group_id"])
            key = hashlib.sha256(json.dumps([config["trainer"].get("seed", 42), meta.get("fingerprint", record)], sort_keys=True).encode()).hexdigest()
            inspection.update(split=split, fingerprint=meta.get("fingerprint"), primary_behavior=meta.get("primary_behavior"))
            for bucket, cap in [("random", 32), *[(tag, 4) for tag in {meta.get("primary_behavior", "unknown"), *meta.get("behavior_tags", [])}]]:
                values = audit_buckets[(split, bucket)]
                values.append((key, inspection))
                values.sort(key=lambda value: value[0])
                del values[cap:]
            if limit is not None and len(rows) >= limit:
                break
        if not rows:
            raise ValueError(f"{split} 没有可用完整样本（超长剔除 {skipped} 条）；不要截断工具链或以测试集替代")
        sets[split] = Dataset.from_list(rows)
        split_groups[split] = groups
        preparation[split] = {"retained": len(rows), "overlength_skipped": skipped,
                              "length_limit": config["data_max_length"], "sample_limit": limit,
                              "sources": dict(sources), "primary_behaviors": dict(behaviors),
                              "skipped_records": skipped_records}
    if split_groups["train"] & split_groups["validation"]:
        raise ValueError("训练与验证的工具签名组重叠")
    audits = {(split, key): detail for (split, _), values in audit_buckets.items() for key, detail in values}
    audit = [detail for _, detail in sorted(audits.items())]
    return sets, audit, preparation


def prepare_episodes(path, split, tokenizer, max_model_length, max_completion_length):
    from .policy import SYSTEM
    from .environment import tool_specs
    rows, families = [], set()
    for raw in read_rows(path):
        episode = EpisodeSpec.model_validate(raw)
        if not episode.family_id or episode.split != split:
            raise ValueError("外部可执行任务必须带有预先指定的 family_id 和正确 split")
        prompt = [{"role": "system", "content": SYSTEM + "\n缺参时使用 user_ask 询问，禁止猜测。"}, {"role": "user", "content": episode.request}]
        prompt_ids = tokenizer.apply_chat_template(prompt, tools=tool_specs(), enable_thinking=False, add_generation_prompt=True, tokenize=True)
        if len(prompt_ids) + max_completion_length > max_model_length:
            raise ValueError("工具 schema 与请求加上轨迹预算超过上下文窗口，请缩短输入或调整配置")
        rows.append({"prompt": prompt, "episode_json": episode.model_dump_json()})
        families.add(episode.family_id)
    if not rows:
        raise ValueError("没有外部可执行任务；当前不生成自建数据，GRPO 无法实际运行")
    return rows, families


def execute(config, mask_only=False):
    require_stack()
    import torch
    from transformers import set_seed
    hardware = config.get("hardware", {})
    cpu_mode = hardware.get("device") == "cpu"
    if not mask_only:
        if cpu_mode:
            torch.set_num_threads(hardware.get("cpu_threads", 4))
        elif not torch.cuda.is_available():
            raise RuntimeError("此配置需要 CUDA；不会自动切换到 CPU")
        elif config["trainer"].get("bf16") and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("当前 GPU 不支持 BF16，请选择 sft-p1000 或 sft-cpu 配置")
        if hardware.get("load_in_4bit"):
            cuda_version = tuple(int(v) for v in torch.version.cuda.split('.')[:2]) if torch.version.cuda else (0, 0)
            if not (11, 8) <= cuda_version <= (12, 6):
                raise RuntimeError("Pascal 配置需要兼容 CUDA 11.8～12.6 的 PyTorch/bitsandbytes 构建；12.8/13.x 预编译 bnb 不包含 Pascal")
            if tuple(torch.cuda.get_device_capability()) < (6, 0):
                raise RuntimeError("NF4 需要 compute capability >=6.0")
            archs = set(torch.cuda.get_arch_list())
            if not ({"sm_60", "sm_61"} & archs):
                raise RuntimeError("当前 PyTorch wheel 没有 Pascal sm_60/sm_61 内核；请更换兼容构建")
            if not importlib.util.find_spec("bitsandbytes"):
                raise RuntimeError("缺少 bitsandbytes；请先在独立训练环境准备兼容 Pascal 的版本")
    set_seed(config["trainer"]["seed"])
    tokenizer, template = load_tokenizer(config)
    output = Path(config["trainer"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if config["stage"] == "sft":
        datasets, audit, preparation = prepare_sft(config, tokenizer, template)
        (output / "mask-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "data-preparation.json").write_text(json.dumps(preparation, ensure_ascii=False, indent=2), encoding="utf-8")
        if mask_only:
            return {"mask_checked": True, "audited_examples": len(audit), "trained": False}
    elif mask_only:
        raise ValueError("mask-only 仅用于 SFT")
    model, lora, targets = load_model_and_lora(config)
    if config["stage"] == "sft":
        from trl import SFTConfig, SFTTrainer
        trainer = SFTTrainer(model=model, args=SFTConfig(**config["trainer"]), processing_class=tokenizer,
            train_dataset=datasets["train"], eval_dataset=datasets["validation"], peft_config=lora)
    else:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
        from .integrations.trl_env import TaskToolEnv
        from .rewards import RewardConfig, recorded_trl_reward
        rows, families = prepare_episodes(config["episodes_file"], "train", tokenizer,
            config["trainer"].get("vllm_max_model_length", 8192), config["trainer"]["max_completion_length"])
        eval_dataset = None
        if config.get("validation_episodes_file"):
            validation, validation_families = prepare_episodes(config["validation_episodes_file"], "validation", tokenizer,
                config["trainer"].get("vllm_max_model_length", 8192), config["trainer"]["max_completion_length"])
            if families & validation_families:
                raise ValueError("可执行训练与验证任务的 family_id 重叠")
            eval_dataset = Dataset.from_list(validation)
        reward_config = RewardConfig(**config["reward"])
        trainer_type = make_budgeted_trainer(GRPOTrainer, config["per_turn_max_new_tokens"])
        trainer = trainer_type(model=model, args=GRPOConfig(**config["trainer"]), processing_class=tokenizer,
            train_dataset=Dataset.from_list(rows), peft_config=lora,
            eval_dataset=eval_dataset,
            reward_funcs=[recorded_trl_reward(output, config["trainer"]["num_generations"])],
            environment_factory=lambda: TaskToolEnv(reward_config=reward_config))
    trainable = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name or "language_model" not in name or "visual" in name for name in trainable):
        raise ValueError("可训练参数超出语言侧 LoRA 范围")
    manifest = {"config": config, "environment": capture_environment(), "lora_targets": targets,
                "chat_template_sha256": hashlib.sha256((template or tokenizer.chat_template).encode()).hexdigest(),
                "command": sys.argv, "status": "started"}
    (output / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cpu_mode:
        torch.cuda.reset_peak_memory_stats()
    before = parameter_fingerprint(trainer.model)
    trainer.train()
    after = parameter_fingerprint(trainer.model)
    if before == after:
        raise RuntimeError("训练结束但 LoRA 参数未变化，禁止报告成功更新")
    (output / "trainer-metrics.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer.save_model(str(output / "final-adapter"))
    tokenizer.save_pretrained(output / "final-adapter")
    if config.get("export_merged"):
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output / "final-merged")
        tokenizer.save_pretrained(output / "final-merged")
    manifest.update(status="completed", peak_memory_bytes=torch.cuda.max_memory_allocated() if not cpu_mode else None,
                    trainable_parameters_changed=True, parameter_hash_before=before, parameter_hash_after=after)
    (output / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"trained": True, "output": str(output)}


def main():
    from .console import utf8_console
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true", help="明确运行训练；默认只输出计划")
    parser.add_argument("--check-mask", action="store_true", help="使用本地模型 tokenizer 检查真实样本，不更新参数")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.execute and args.check_mask:
        parser.error("execute 与 check-mask 互斥")
    result = execute(config, mask_only=args.check_mask) if args.execute or args.check_mask else plan(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
