# 项目目录导航

日常开发以八工具任务环境、真实模型接入和自动评测为主；SFT/GRPO 代码与数据保留为可选实验模块。现有旧环境与八工具环境的协议不同，不能直接混接。

## 代码与入口

| 路径 | 用途 |
| --- | --- |
| `wearable_tasks/` | 当前八工具模拟环境、任务生成、终局判定、重放和负例审计 |
| `wearable_tasks/tests/` | 八工具环境的分阶段单元测试 |
| `src/wearable_assistant/` | 当前固定模型运行器、三组评测、公开状态/执行校验、本地模型服务及 Web/API |
| `src/lightagent/` | 原工具环境、模型 HTTP 接口、运行器、公开数据准备和 TRL 训练入口；部分模块仍用于当前数据流程 |
| `tests/` | 原环境及当前统一数据、模板、训练配置的回归测试 |
| `configs/` | 数据处理与实验配置；具体选择见下文 |
| `scripts/` | 数据准备、下载、设备检查、受控试跑和清理脚本 |

当前八工具推理/批量评测接入已在 `wearable_assistant` 中实现，运行方法见 [助手系统](assistant-system.md)。真实模型服务与完整能力评测仍需手动运行。`ReferenceEnvironmentLoader` 本身仍是框架无关加载器，不等同于 GRPO Trainer 训练适配器。

## 数据与模型

| 路径 | 用途及保留原因 |
| --- | --- |
| `data/wearable_tasks/formal/` | 2,900 条正式任务：训练 2,000、验证 300、三类测试各 200；`private/` 仅供环境与评分器 |
| `data/wearable_tasks/smoke/` | 100 条任务构造冒烟数据及报告，也是正式数据重建时的前置证据 |
| `data/raw/` | 原始公开数据、许可证与下载清单，用于来源追溯和重新处理 |
| `data/sft_v1_inventory/` | 当前有效、长度已修正的完整数据池，用于重新抽样及核对供给，保留 |
| `data/sft_v1/` | 已抽样的正式统一数据和全部独立元数据；模型输入只含 tools/messages |
| `models/Qwen3.5-2B/` | 固定 revision 的完整模型权重、配置与 tokenizer，可用于推理和实验 |

统一数据含 9,000 条训练、700 条验证与三类各 200 条测试记录。4096-token 训练视图会整条跳过原有 3 条超长记录，实际使用 8,997 条训练；`local_smoke/` 的 383/23 条仅用于短样本流程。

私有目标、故障计划和用户模拟答案不能拼入模型消息。正式数据目录下的元数据、报告及 mask 审计存在哈希依赖，不要只复制 train/validation 两个 JSONL。

## 配置选择

- 数据准备：`sft-data-v1.json`、`sft-mixture-v1.json`；完整说明见 [数据准备](sft-data-v1.md)。
- 固定模型助手：`assistant.json`；控制器、API、评测无需训练。`run_assistant.ps1` 默认预览，显式 `-Execute` 才发起真实推理或启动服务。
- 当前统一数据 SFT：`sft-v1-full.json`（BF16）、`sft-v1-p1000-full.json`（完整视图 NF4）、`sft-v1-p1000.json`（20 步冒烟）、`sft-v1-p1000-train.json`（短样本单轮）。
- 历史 SFT：`sft.json`、`sft-a100.json`、`sft-p1000.json`、`sft-cpu.json` 指向旧数据或旧入口；保留供代码/历史实验追溯，不作为当前正式数据训练指令。
- `grpo.json` 与 `experiments.json` 仍是未接通的旧实验方案，不能直接启动整套正式实验；缺口见 [就绪报告](../reports/remote-experiment-readiness.json)。

## 运行环境和结果

- `.venv/`：原 Python 3.10 轻量运行/测试环境。
- `.venv-data/`：Python 3.11 数据处理环境。
- `.venv-train/`：本机 CUDA/TRL 训练环境；推理与训练实验仍可能使用，保留。
- `artifacts/`：已有 adapter、训练记录、mask 审计和长文本试跑证据，保留。
- `reports/`：数据、环境、设备、实验就绪与本次清理报告。
- `outputs/`：工程回归及任务工厂的测试记录；其中部分报告是重建前置条件，保留。

最新核验结果见 [长文本报告](../reports/p1000-long-context-report.md) 与 [训练说明](sft-training-v1.md)。历史报告中的“尚未生成/尚未训练”等描述以其生成时间为准，不代表当前总状态。

## 本次清理及后续维护

清理目标限定为已验证模型下载完成后的中断下载缓存、明确标记无效的旧长度数据池，以及项目代码目录中的 Python/pytest 缓存。采用 Windows 回收站，不永久删除，不处理其他项目或全局缓存。

- 清理收据：[workspace-cleanup.json](../reports/workspace-cleanup.json)。
- 无效旧盘点的报告与 tokenizer 信息：`reports/archive/invalid-lengths-20261116/`。
- 旧报告中出现的 `data/sft_v1_inventory.invalid-lengths-20261116` 仅为历史来源；清理后不能再作为命令输入。当前有效数据池为 `data/sft_v1_inventory/`。
- 移到回收站后可恢复；只有之后清空回收站才实际释放磁盘空间。本脚本不会清空回收站。

从项目根目录使用 PowerShell 7：

```powershell
# 仅预览固定范围，不清理文件
& './scripts/cleanup_workspace.ps1'

# 核验模型/数据哈希、归档旧报告后，将固定范围送入回收站
& './scripts/cleanup_workspace.ps1' -Recycle
```

脚本不会清理虚拟环境、模型权重、原始数据、有效盘点池、正式任务数据或已有 adapter。项目 Python/pytest 缓存会在以后运行时自动重新生成。
