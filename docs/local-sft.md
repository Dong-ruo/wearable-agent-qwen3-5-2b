# P1000 4GB 本地 SFT 参数

当前统一数据请使用 `configs/sft-v1-p1000.json`；完整 SFT 使用 `configs/sft-v1-full.json`，4096-token 上限整条跳过 3 条后实际读取 8,997/700。最新参数核对、独立 CUDA 环境和可复制训练指令见 [SFT 训练指令](sft-training-v1.md)。本页其余旧配置命令保留作历史参考。

P1000 上的 512-token、NF4/FP32 冒烟已完成 20 步，PyTorch 峰值分配显存约 3.38GiB，LoRA 参数发生更新。新增 `configs/sft-v1-p1000-train.json` 使用全部 383/23 条短样本运行 1 epoch；新配置尚未开始训练。长上下文完整数据的 4GB 适配未验证，CPU 模式在 16GB 内存内的可行性也未验证。

## 配置选择

| 配置 | 设备/精度 | 上限 | 用途 |
| --- | --- | --- | --- |
| configs/sft.json / sft-p1000.json | P1000，NF4 4-bit，FP32 计算 | 512 token，128 条训练/16 条验证，20 步 | 默认的低显存试运行 |
| configs/sft-cpu.json | CPU，FP32，4 线程 | 256 token，32 条训练/4 条验证，2 步 | 手动选择的 CPU 功能检查 |
| configs/sft-a100.json | 原 BF16 LoRA 配置 | 原 4096 token / rank 16 | 保留原配置，不用于 P1000 |

P1000 配置关闭 BF16/FP16 AMP；使用标准 AdamW，不用 FlashAttention、fused optimizer 或 vLLM。LoRA 仅适配语言侧 q_proj/v_proj，rank=4、alpha=8，减小优化器与梯度开销。Qwen 的线性注意力模块没有因此全部被适配，不能把这份冒烟设置当成质量最优配方。

QLoRA 加载时使用 BitsAndBytesConfig，开启 NF4 与 double quant，再执行 prepare_model_for_kbit_training。整个模型明确加载到 GPU 0；不自动 offload、不自动回退 CPU、不合并量化基础模型。若 OOM，应停止并显式选择 CPU 配置或更换硬件，不能宣称已经适配成功。

## 环境前提

1. 当前旧 .venv 是 Python 3.10；TRL 1.18.0 需要独立 Python >=3.11 训练环境。不要沿用标准库任务工厂的 `-S` 参数启动训练，训练需要正常加载安装的依赖。
2. 先准备支持 Pascal sm_60/sm_61 的 PyTorch 构建，并与 bitsandbytes 的 CUDA 11.8～12.6 后端匹配。驱动 nvidia-smi 显示的 CUDA 上限不是 torch.version.cuda，也不能证明 wheel 含 Pascal 内核。
3. bitsandbytes 官方表格中，NF4 最低 CC 6.0；但 CUDA 12.8/13 系列预编译包不包含 Pascal。不要无条件使用最新 PyTorch/CUDA/bitsandbytes；某些旧 CUDA wheel 与本项目新 TRL/Transformers 组合也可能没有兼容版本，需要实际解析依赖后确认。入口会拒绝明显不兼容的 CUDA/架构组合，仍不能替代真实内核测试。
4. 已在 `.venv-train` 安装 CUDA 11.8 的 PyTorch 2.6.0 与 bitsandbytes 0.49.0，基础 NF4 前向/反向通过；实际软件版本保存在 `reports/sft-training-environment.txt`。这不是完整 Qwen 训练已通过的结论。

官方依据：[bitsandbytes 安装与架构表](https://huggingface.co/docs/bitsandbytes/installation)、[Transformers 4-bit 训练说明](https://huggingface.co/docs/transformers/quantization/bitsandbytes)。

## 数据与长度

旧配置仍指向 `data/public/train.jsonl`、`validation.jsonl`，使用旧公开数据契约。统一转换、分组划分、真实长度盘点和独立元数据已由新 `lightagent.sft_data` 入口接入，手动执行说明见 [统一 SFT 数据准备](sft-data-v1.md)。新候选配置 `configs/sft-v1-p1000.json` 读取 `data/sft_v1/local_smoke/`，并检查独立元数据、文件哈希及当前 mask 审计；旧配置不自动切换。新入口已实际读取 128 条训练、16 条验证的短样本并编码，未创建 Trainer。

低显存模式按真实 tokenizer 的完整轨迹长度筛掉超长记录，不从中间截断调用/返回链。512/256 token 可能容纳不了多工具 schema 或多数长轨迹；筛选后为空会明确报错，必须准备真实的短完整样本，不能删除工具反馈来“凑长度”。保留数与超长剔除数写入 data-preparation.json。

128/16 的上限是为显存与速度试验设置的小子集，不是正式训练分布；验证分组检查、assistant mask 检查和真实参数变化检查仍保留。20 步配置不会进行周期评测或依据验证 loss 选择最佳检查点。

## 由用户执行的指令

在准备好的 Python >=3.11 训练环境中执行。当前不运行这些命令：

```powershell
# 只看配置计划，无模型加载或训练
python -m lightagent.training --config configs/sft-p1000.json

# 准备好本地模型与转换数据后，先检查真实 mask
python -m lightagent.training --config configs/sft-p1000.json --check-mask

# 兼容性确认后，手动启动 20 步试运行
python -m lightagent.training --config configs/sft-p1000.json --execute

# 明确选择 CPU 回退（不会自动启动），仅两步
python -m lightagent.training --config configs/sft-cpu.json --execute
```

模型加载始终 local_files_only=True。保存 final-adapter、mask-audit.json、data-preparation.json、trainer-metrics.json 与 run-manifest.json，不导出完整合并权重。CPU 的 peak_memory_bytes 保留 null，不会伪造 GPU 峰值或声称已测 CPU 内存。

先用真实试运行测每步时间与峰值显存，确认有参数更新，再决定是否增加长度、样本量与更新步数。本机全量训练时间目前仍未知。
