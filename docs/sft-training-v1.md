# SFT 参数核对与训练指令

独立 `.venv-train` 环境与固定 revision 的模型权重已准备完成。P1000 上的 20 步冒烟训练已实际完成，LoRA 参数变化，耗时约 18.6 分钟，PyTorch 峰值分配显存约 3.38GiB。该数值不含全部 CUDA 分配器保留量和其他应用占用，不能视为可用显存余量。环境记录见 `reports/sft-training-setup.json`，实际训练结果见 `artifacts/sft-v1-p1000-smoke/run-manifest.json`。

## 当前 P1000 的一轮短样本训练

新增 `configs/sft-v1-p1000-train.json`，沿用已实际跑通的模型加载和显存相关参数：NF4 double quant、FP32 计算、512-token 完整协议、micro batch 1、累积 8、gradient checkpointing、LoRA r=4/alpha=8，仅适配 12 个语言侧 q/v 层。禁用 BF16、FP16、packing、torch.compile 与合并导出。

训练范围由原先的 128/16 冒烟子集改为短视图的全部 **383 条训练、23 条验证**，去掉 20 步上限，运行 1 epoch，预计 **48 个优化器步**。学习率 5e-5，linear 调度，warmup 2 步，梯度裁剪 1.0。epoch 结束验证 loss 并保存，验证 micro batch 为 1，仅返回 loss，避免积累整批 logits。48 项测试通过，全部短样本已重新编码并通过角色 mask 检查；尚未执行这轮新训练。

这仍是本机短样本视图，全部来自公开数据；长于 512 的自建轨迹不在其中。完整 8,997 条多轮 SFT 仍保留给更大显存设备，不删 schema、反馈或用户消息来塞入 P1000，也不把这一轮称为完整多轮训练。旧完整配置与原始数据保持不变。

在项目根目录的 PowerShell 中执行以下命令才会启动新训练：

```powershell
& '.\.venv-train\Scripts\python.exe' -X utf8 -m lightagent.training --config configs/sft-v1-p1000-train.json --execute
```

从原始基础权重开始，结果保存到独立目录 `artifacts/sft-v1-p1000-train-seed42/`，不会覆盖已经完成的冒烟 adapter。按本机已完成冒烟粗估，单轮训练约 45 分钟，验证与保存另计，实际随序列长度和设备负载变化。调整依据及数据计数见 `reports/sft-p1000-local-settings.json`。

## 已核对的配置

| 项目 | 本机冒烟 | 完整 SFT |
| --- | --- | --- |
| 配置 | `configs/sft-v1-p1000.json` | `configs/sft-v1-full.json` |
| 实际读取训练/验证 | 128 / 16 | 8,997 / 700 |
| 长度上限 | 512 | 4096 |
| 超长处理 | 数据视图已筛好，不截断 | 整条舍弃 3 条公开反馈轨迹 |
| 精度 | NF4 4-bit，FP32 计算 | BF16 LoRA |
| micro batch / 梯度累积 | 1 / 8 | 1 / 16 |
| 单 GPU 有效 batch | 8 | 16，末批可能较小 |
| 学习率 | 5e-5，linear | 1e-4，cosine |
| warmup / weight decay | 0 步 / 0 | 17 步（约 3%） / 0.01 |
| LoRA r / alpha / dropout | 4 / 8 / 0 | 16 / 32 / 0.05 |
| LoRA 范围 | 12 个语言侧 q/v 层 | 186 个语言侧层，含 90 个线性注意力投影 |
| 训练量 | 20 个优化器步 | 1 epoch，单 GPU 预计 563 个优化器步 |
| 验证/保存 | 不周期验证；结束保存 adapter | 每个 epoch 验证和保存；结束保存 adapter |
| seed / data_seed | 42 / 42 | 42 / 42 |

按用户要求，完整训练保持 4096，不为了 3 条超长记录提高上限。过滤发生在加载时，不改变已准备数据与元数据的追溯关系；实际保留自建训练 2,000、公开训练 6,997，验证仍为自建 300＋公开 400。三类测试各 200 条不参与训练或选模。

20 步是优化器更新数，使用 128 条训练样本、累积 8 时约使用 160 次样本，会进入第二次遍历；`max_steps` 优先于 `num_train_epochs=1`。这只是链路检查。完整 SFT 不设置正数 `max_steps`，按一个 epoch 运行。

两套配置均保留 gradient checkpointing、`assistant_only_loss=True`、显式 `labels`，屏蔽 system/user/tool，关闭 packing 和自动截断。显式对齐 EOS 到 `<|im_end|>`。当前 Transformers 使用 `warmup_steps`，完整配置设为 17 步，不传已移除的 `warmup_ratio`。完整实验只跑一个 epoch，因此保留最终权重，不声称进行过多检查点择优。不会在线合并基础权重。

修正了 TRL 1.18.0 跳过预处理时需要 `labels` 和对话格式标识的问题，并避免 Arrow 将不同工具的同名参数强制转换成相同类型。附加对话列只供 TRL 格式识别，collator 仅传入 `input_ids/attention_mask/labels`；完整工具协议已保存在编码中，原始 JSONL 不变。

本次验证：相关测试现为 45 项通过，包含两份实际配置的 TRL 参数解析和 BF16 软件模拟防误判；实际 TRL 构造器与 collator 使用随机微型模型检查通过，未调用 forward/train；完整 8,997/700 和冒烟 128/16 均已实际编码并完成角色 mask 检查。LoRA 目标在 meta 模型结构上核对，未加载 Qwen 权重。参数核对见 `reports/sft-settings-review.json`，最新环境执行结果见 `reports/sft-training-setup.json`。

## 当前环境的限制

本机为 Quadro P1000 4GB、CC 6.1，约 16GB 内存。P1000 不适用 BF16 完整配置；NF4/FP32、512-token 的 20 步冒烟已完成。2026-11-17 另用真实完整轨迹完成长文本前向/反向试跑，最长 3816 token；运行依赖 Windows 共享 GPU 内存。整轮完整数据训练尚未执行，具体通过次数、耗时和测试限制见 `reports/p1000-long-context-report.json`。

`.venv-data` 保留为 `torch==2.15.0+cpu` 的数据环境；训练使用独立 `.venv-train`，实际安装 `torch==2.6.0+cu118`、TRL 1.18.0、Transformers 5.23.0、PEFT 0.21.3、bitsandbytes 0.49.0。checkpoint 权重已补齐。旧 `sft.json` / `sft-p1000.json` / `sft-a100.json` 指向旧数据，旧 CPU 配置的 256-token 上限也容纳不了本次有效短样本，不用于当前流程。

以下 P1000 组合已在实际 CC 6.1 GPU 上通过基础 CUDA/NF4 内核检查和 512-token 的 20 步训练；长轨迹的受控测试见下文，不能代替整轮稳定性和能力评估。驱动显示的 CUDA 12.8 不等于 PyTorch 自带的 11.8 运行时。依据：[PyTorch 旧版本安装命令](https://docs.pytorch.org/get-started/previous-versions/)、[bitsandbytes 0.49.0 平台与架构表](https://huggingface.co/docs/bitsandbytes/v0.49.0/installation)。BF16 检查明确排除软件模拟，P1000 不符合完整 BF16 配置。

## 1. 手动准备独立 P1000 环境

环境安装与权重下载已执行。下面保留同一个 PowerShell 窗口中的复现命令；后面的 `--execute` 训练命令尚未执行。

```powershell
Set-Location -LiteralPath 'D:\ms-swift＋Qwen3.5-2B'
$ErrorActionPreference = 'Stop'
py -3.11 -m venv .venv-train
if ($LASTEXITCODE -ne 0) { throw '创建训练环境失败' }
$TrainPython = (Resolve-Path '.\.venv-train\Scripts\python.exe').ProviderPath
function Invoke-TrainPython {
    & $TrainPython @args
    if ($LASTEXITCODE -ne 0) { throw "命令失败，退出码 $LASTEXITCODE；停止后续操作" }
}
Invoke-TrainPython -X utf8 -m pip install --upgrade pip
Invoke-TrainPython -X utf8 -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu118
Invoke-TrainPython -X utf8 -m pip install -e '.[dev]' -r requirements/sft-p1000.txt
Invoke-TrainPython -X utf8 -m pip check

# 检查 CUDA、Pascal 内核及一个微型 NF4 前向/反向；不加载 Qwen、不更新模型参数
Invoke-TrainPython -X utf8 scripts/check_sft_cuda.py --config configs/sft-v1-p1000.json
```

预检查失败时停止，不自动换成 CPU 或修改全局环境。NF4 微型内核通过也不能证明完整模型显存足够。

## 2. 手动补齐固定 revision 的模型权重

只补权重和可选 generation config，不覆盖已审计 tokenizer。会下载约 4.55GB checkpoint。

```powershell
Invoke-TrainPython -X utf8 scripts/download_sft_weights.py --http --workers 4
```

## 3. 手动启动本机 20 步冒烟

最后一条 `--execute` 才会真正启动训练。

```powershell
Invoke-TrainPython -X utf8 scripts/check_sft_cuda.py --config configs/sft-v1-p1000.json --require-weights
Invoke-TrainPython -X utf8 -m lightagent.training --config configs/sft-v1-p1000.json --check-mask
Invoke-TrainPython -X utf8 -m lightagent.training --config configs/sft-v1-p1000.json --execute
```

成功时检查 `artifacts/sft-v1-p1000-smoke/run-manifest.json` 的 `status=completed`、`trainable_parameters_changed=true`、峰值显存和 `trainer-metrics.json`；LoRA 在 `final-adapter/`。若 OOM 或 CUDA 错误，保存错误信息，不把配置计划、mask 通过或目录存在当成训练完成。

## 4. 在合适 GPU 上运行完整 SFT

在支持 BF16、显存足够的 GPU 环境中准备同样版本的 Transformers / TRL / PEFT 与完整 checkpoint。此配置不使用 bitsandbytes，不适用于本机 P1000。软件版本应与数据 tokenizer 审计一致。

```powershell
Invoke-TrainPython -X utf8 scripts/check_sft_cuda.py --config configs/sft-v1-full.json --require-weights
Invoke-TrainPython -X utf8 -m lightagent.training --config configs/sft-v1-full.json --check-mask
Invoke-TrainPython -X utf8 -m lightagent.training --config configs/sft-v1-full.json --execute
```

完整训练输出在 `artifacts/sft-v1-full-seed42/`。训练加载报告会列出整条跳过的 3 条记录及来源 ID。

## 5. P1000 完整数据配置与长轨迹验证

`configs/sft-v1-p1000-full.json` 使用完整训练文件，4096-token 上限下整条跳过原有 3 条超长记录，保留 8,997 条训练记录（含全部 2,000 条自建轨迹）及 700 条验证记录。它没有样本数量上限，也没有自动截断。当前保留的训练记录最长为 3816 token，验证记录最长为 2602 token；这不等同于已验证实际 4096、8K 或更长输入。

模型内存设置采用实际试跑的 NF4、FP32 compute、eager attention、rank 4/alpha 8 的语言侧 q/v LoRA、batch size 1 和非重入梯度检查点。CPU 激活卸载的对照试跑使用了更多共享 GPU 内存，因此本配置关闭该选项。长度包含原有 system、工具定义、全部消息及返回。公开记录的本次试跑仅检查模型计算，没有重新执行公开 API。

整轮计划为梯度累积 8、一个 epoch、约 1,125 次优化器更新，学习率 5e-5、线性调度、预热 34 步；每 50 步保存 checkpoint，最多保留两个，末尾做一次完整验证并保存最终 adapter。这些整轮设置已整理成配置，但整轮训练尚未执行。新输出目录为 `artifacts/sft-v1-p1000-full-seed42/`，与已有冒烟及短样本训练目录区分。

受控测试使用真实 TRL `training_step`、实际 loss/梯度校验并预先分配 Adam 状态。较长测试的两次重复合计超过 6 分钟预算时被终止，报告保留已完成的前向/反向次数，并明确标记超时；超时不视为 CUDA OOM。最终单次更新的结果也写入报告。试跑不会保存 adapter 或启动整轮 SFT。

```powershell
# 复查完整数据和 mask，不加载模型权重，不启动训练
& '.\.venv-train\Scripts\python.exe' -X utf8 -m lightagent.training --config configs/sft-v1-p1000-full.json --check-mask
if ($LASTEXITCODE -ne 0) { throw '完整数据 mask 检查失败' }

# 仅在准备正式长时间训练时手动执行；本次验证没有运行此命令
& '.\.venv-train\Scripts\python.exe' -X utf8 -m lightagent.training --config configs/sft-v1-p1000-full.json --execute
if ($LASTEXITCODE -ne 0) { throw '完整数据 SFT 失败' }
```

重新执行受控试跑需使用新的输出目录；以下只在最后一条轨迹做一次临时优化器更新，不保存 adapter：

```powershell
$ProbeOutput = 'artifacts/p1000-context-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& '.\.venv-train\Scripts\python.exe' -X utf8 scripts/probe_sft_context.py --limits 1024 2048 4096 --repeats 1 --max-seconds 600 --output $ProbeOutput
if ($LASTEXITCODE -ne 0) { throw '受控测试执行异常' }
Get-Content -LiteralPath (Join-Path $ProbeOutput 'summary.json') -Raw -Encoding UTF8
```

控制器退出码只说明测试流程是否完成，实际通过、OOM、超时或低内存保护结果需查看 `summary.json` 的每个 trial。整轮内存稳定性和训练效果仍需正式运行后判断；长文本单条运行已达到数分钟，不能按 512-token 冒烟速度估算整轮耗时。

## 实验解释与后续比较

- 先保留固定 revision 的基础模型结果，再比较 SFT adapter；对照使用相同提示模板、工具定义、解码参数和测试任务。
- 验证 loss 用于观察训练过程；三类自建测试仍各 200 条，用工具执行正确性、终局奖励、澄清和错误恢复等指标评估能力，不能把 loss 下降直接等同于工具成功率提高。
- 首轮完整 SFT 使用一个 seed、一个 epoch，只能作为首轮基线。需要稳健结论时再用 42/43/44 等种子重复；不要把单次运行称为稳定提升。
- 公共数据没有真实 API 重执行证明。公开无调用示范仍缺失，不宣称已学会该行为。512-token 冒烟不包含自建轨迹，不代表完整多轮任务的学习结果。
