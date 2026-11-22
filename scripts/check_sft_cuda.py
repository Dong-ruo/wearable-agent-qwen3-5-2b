"""Check CUDA/quantization prerequisites; never load Qwen or start training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--output", default="reports/sft-cuda-preflight.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    import torch
    errors, packages = [], {}
    for name, expected in (("trl", "1.18.0"), ("transformers", "5.23.0"), ("peft", "0.21.3")):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
        if packages[name] != expected:
            errors.append(f"requires {name}=={expected}; found {packages[name]}")
    if config.get("hardware", {}).get("load_in_4bit"):
        try:
            packages["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError:
            packages["bitsandbytes"] = None
            errors.append("bitsandbytes is missing")
    result = {"config": args.config, "packages": packages, "torch": torch.__version__,
              "python_executable": sys.executable, "python_version": sys.version,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "qwen_weights_loaded": False, "training_started": False}
    if not result["cuda_available"]:
        errors.append("CUDA unavailable: this environment cannot run either supplied GPU SFT profile")
    else:
        result.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                      arch_list=torch.cuda.get_arch_list(), free_total_bytes=torch.cuda.mem_get_info(0),
                      bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False),
                      bf16_emulation_supported=torch.cuda.is_bf16_supported())
        if config["trainer"].get("bf16") and not torch.cuda.is_bf16_supported(including_emulation=False):
            errors.append("full SFT profile requires native BF16 support")
        if config.get("hardware", {}).get("load_in_4bit"):
            version = tuple(int(v) for v in torch.version.cuda.split(".")[:2]) if torch.version.cuda else (0, 0)
            if not (11, 8) <= version <= (12, 6):
                errors.append("P1000 profile requires CUDA 11.8 through 12.6")
            if not {"sm_60", "sm_61"}.intersection(result["arch_list"]):
                errors.append("PyTorch wheel has no Pascal sm_60/sm_61 kernels")
            if not errors:
                try:
                    import bitsandbytes as bnb
                    layer = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.float32,
                                              quant_type="nf4", compress_statistics=True).cuda()
                    x = torch.randn(2, 32, device="cuda", requires_grad=True)
                    loss = layer(x).float().square().mean()
                    loss.backward()
                    torch.cuda.synchronize()
                    if x.grad is None or not torch.isfinite(x.grad).all().item():
                        raise ValueError("missing/nonfinite input gradient")
                    result["tiny_nf4_forward_backward"] = "passed_no_optimizer_or_qwen_model"
                except Exception as exc:
                    errors.append("NF4 CUDA kernel check failed: " + str(exc))
    if args.require_weights:
        directory = Path(config["model_name_or_path"])
        index_path = directory / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            required = sorted(set(index["weight_map"].values()))
        else:
            required = ["model.safetensors"]
        missing = [f for f in required if not (directory / f).is_file() or (directory / f).stat().st_size == 0]
        result["missing_weight_files"] = missing
        if missing:
            errors.append("model checkpoint is incomplete; tokenizer files alone are insufficient")
    result.update(status="blocked" if errors else "passed", errors=errors,
                  scope="Prerequisites only; full-model memory fit and training remain untested")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
