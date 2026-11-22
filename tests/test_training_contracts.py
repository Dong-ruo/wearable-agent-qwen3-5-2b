import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from lightagent.benchmarks import benchmark_plan
from lightagent.data_pipeline import convert_toolace
from lightagent.training import assert_mask, encode_record, load_config, make_budgeted_trainer, plan, turn_budget
from test_data_pipeline import toolace


def test_all_training_defaults_are_plans_and_do_not_need_a_gpu():
    for name in ("sft", "grpo"):
        result = plan(load_config(f"configs/{name}.json"))
        assert not result["executed"]
        assert result["framework"] == "trl"
        assert result["missing_local_paths"]
    cfg = load_config("configs/sft.json")
    cfg["trainer"]["assistant_only_loss"] = False


class FakeNativeTokenizer:
    """Tests label accounting only; deliberately not presented as Qwen template validation."""
    leak = False

    def apply_chat_template(self, messages, tokenize=True, return_dict=False, **kwargs):
        rendered, mask = "", []
        for message in messages:
            segment = message["role"] + ":" + (message.get("content") or "") + json.dumps(message.get("tool_calls", [])) + "|"
            rendered += segment
            enabled = int(message["role"] == "assistant" or (self.leak and message["role"] == "tool"))
            mask.extend([enabled]*len(segment))
        if not tokenize:
            return rendered
        return {"input_ids": [ord(c) for c in rendered], "assistant_masks": mask}

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_sft_masks_calls_and_answers_but_never_user_or_tool_observations():
    record = convert_toolace(toolace())
    encoded, report = encode_record(FakeNativeTokenizer(), record, "fake", 10000)
    for segment in report["segments"]:
        assert (segment["supervised_tokens"] > 0) == (segment["role"] == "assistant")
    assert "Get_Weather" in report["segments"][2]["supervised_text"]
    assert "查询返回下雨" in report["segments"][-1]["supervised_text"]
    assert -100 in report["labels"]
    assert encoded["labels"] == report["labels"]
    assert all(label == (token if keep else -100) for token, keep, label in
               zip(encoded["input_ids"], encoded["assistant_masks"], encoded["labels"]))
    tokenizer = FakeNativeTokenizer()
    tokenizer.leak = True
    with pytest.raises(ValueError, match="未屏蔽"):
        encode_record(tokenizer, record, "fake", 10000)
    with pytest.raises(ValueError, match="超长"):
        encode_record(FakeNativeTokenizer(), record, "fake", 20)
    with pytest.raises(ValueError, match="mask"):
        assert_mask([1, 2], [0, 0])


@pytest.mark.parametrize("use_vllm", [True, False])
def test_per_turn_cap_is_separate_from_trajectory_budget_and_restored(use_vllm):
    owner = SimpleNamespace(max_completion_length=2048, max_new_tokens=2048)
    trainer = SimpleNamespace(use_vllm=use_vllm, vllm_generation=owner, generation_config=owner, max_completion_length=2048)
    attribute = "max_completion_length" if use_vllm else "max_new_tokens"
    with pytest.raises(RuntimeError):
        with turn_budget(trainer, 256):
            assert getattr(owner, attribute) == 256
            assert trainer.max_completion_length == 2048
            raise RuntimeError("interrupted")
    assert getattr(owner, attribute) == 2048


def test_budgeted_grpo_subclass_rejects_an_ineffective_upstream_cap():
    class FakeTrainer:
        use_vllm = False
        generation_config = SimpleNamespace(max_new_tokens=2048)

        def _generate_single_turn(self, *args):
            return [[0]*300], None, None
    trainer = make_budgeted_trainer(FakeTrainer, 256)()
    with pytest.raises(RuntimeError, match="预算没有生效"):
        trainer._generate_single_turn([], None, {}, 4)


def test_benchmark_smoke_and_formal_scope_are_never_labelled_total_score():
    cfg = json.loads(Path("configs/bfcl-smoke.json").read_text())
    smoke = benchmark_plan(cfg)
    formal = benchmark_plan(cfg, formal=True)
    assert not smoke["executed"] and smoke["sample_counts"] is None
    assert smoke["config"]["limit"] == 5
    assert "limit" not in formal["config"]
    assert "非 BFCL 总分" in formal["scope"]
    assert cfg["limit"] == 5


@pytest.mark.parametrize("module,args", [
    ("lightagent.training", ["--config", "configs/sft.json"]),
    ("lightagent.training", ["--config", "configs/grpo.json"]),
    ("lightagent.benchmarks", ["--config", "configs/bfcl-smoke.json"]),
])
def test_default_commands_emit_valid_utf8_json_without_starting_experiments(module, args):
    process = subprocess.run([sys.executable, "-m", module, *args], capture_output=True, encoding="utf-8", check=True)
    assert json.loads(process.stdout)["executed"] is False


def test_plan_distinguishes_tokenizer_directory_from_weight_checkpoint(tmp_path):
    config = load_config("configs/sft-v1-p1000.json")
    config["model_name_or_path"] = str(tmp_path)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert plan(config)["local_model_weights_present"] is False


def test_real_trl_accepts_preencoded_assistant_labels_without_training(tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("transformers")
    if not Path("models/Qwen3.5-2B/tokenizer.json").exists():
        pytest.skip("actual local Qwen tokenizer unavailable")
    from datasets import Dataset
    from transformers import GPT2Config, GPT2LMHeadModel
    from trl import SFTConfig, SFTTrainer
    from lightagent.training import load_tokenizer, trainer_row
    import torch
    tokenizer, template = load_tokenizer({"model_name_or_path": "models/Qwen3.5-2B"})
    record = convert_toolace(toolace())
    encoded, _ = encode_record(tokenizer, record, template, 4096)
    row = trainer_row(encoded, record)
    # Random tiny fixture: constructor/collator compatibility only, no train() or forward().
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=4096, n_embd=8, n_layer=1, n_head=1))
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        assistant_only_loss=True, completion_only_loss=False, packing=False, max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True}, report_to="none", dataloader_pin_memory=False)
    trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer, train_dataset=Dataset.from_list([row]))
    batch = trainer.data_collator([row])
    assert torch.equal(batch["labels"][0], torch.tensor(encoded["labels"]))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


def test_auxiliary_conversations_do_not_force_arrow_parameter_types():
    pytest.importorskip("datasets")
    from datasets import Dataset
    from lightagent.training import trainer_row
    record = convert_toolace(toolace())
    other = copy.deepcopy(record)
    other["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"city": 7}
    encoded = {"input_ids": [1, 2], "labels": [-100, 2], "assistant_masks": [0, 1], "attention_mask": [1, 1]}
    rows = [trainer_row(encoded, r) for r in (record, other)]
    dataset = Dataset.from_list(rows)
    assert len(dataset) == 2
    assert rows[0]["input_ids"] == encoded["input_ids"]
    assert "tool_calls" in record["messages"][2]
    assert "tool_calls" not in rows[0]["messages"][2]


@pytest.mark.parametrize("name", ["sft-v1-p1000", "sft-v1-full", "sft-v1-p1000-train", "sft-v1-p1000-full"])
def test_actual_sft_configs_match_pinned_trl_argument_schema(name):
    pytest.importorskip("trl")
    from trl import SFTConfig
    config = load_config(f"configs/{name}.json")
    kwargs = copy.deepcopy(config["trainer"])
    kwargs.update(use_cpu=True, bf16=False, fp16=False)
    args = SFTConfig(**kwargs)
    assert args.assistant_only_loss
    assert args.max_length is None
    assert args.warmup_steps == {"sft-v1-p1000": 0, "sft-v1-full": 17, "sft-v1-p1000-train": 2, "sft-v1-p1000-full": 34}[name]


def test_full_gpu_profile_rejects_emulated_bf16_before_loading_model(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from lightagent.training import execute
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: including_emulation)
    with pytest.raises(RuntimeError, match="BF16"):
        execute(load_config("configs/sft-v1-full.json"))


def test_environment_capture_accepts_localized_non_utf8_diagnostics(monkeypatch):
    from lightagent.training import capture_environment
    monkeypatch.setattr("lightagent.training.shutil.which", lambda _: "nvidia-smi")
    monkeypatch.setattr("lightagent.training.platform.platform", lambda: "Windows-test")
    def run(command, **kwargs):
        assert kwargs["capture_output"]
        assert not kwargs.get("text") and "encoding" not in kwargs
        return SimpleNamespace(stdout=b"Quadro P1000, 4096 MiB, 573.73\n",
                               stderr="无法查询设备".encode("gbk"), returncode=1)
    monkeypatch.setattr("lightagent.training.subprocess.run", run)
    result = capture_environment()
    assert result["gpu"].startswith("Quadro P1000")
    assert isinstance(result["gpu_query_stderr"], str) and result["gpu_query_stderr"]
    assert result["gpu_query_returncode"] == 1


def test_device_local_training_uses_complete_short_view_for_one_epoch():
    config = load_config("configs/sft-v1-p1000-train.json")
    assert not config.get("sample_limits")
    assert config["trainer"]["max_steps"] == -1
    assert config["trainer"]["num_train_epochs"] == 1
    assert config["trainer"]["prediction_loss_only"]
    assert config["trainer"]["output_dir"] != load_config("configs/sft-v1-p1000.json")["trainer"]["output_dir"]
