# 面向智能穿戴场景的多轮工具调用 Agent

目标是完成 **查询 → 判断 → 执行 → 核验**，以最终状态和准确反馈衡量效果。

第一次阅读项目，建议先看[项目全程梳理](docs/project-walkthrough.md)：从最初目标、方法和实现阶段，到数据/硬件问题、补充与纠偏，以及当前哪些结果已验证、哪些仍待实测。

目录用途、当前与历史配置的区别，以及清理范围见[项目目录导航](docs/project-layout.md)。训练模块保留为可选实验内容；后续可先围绕固定模型的工具执行、状态管理和自动评测完善系统。

当前已有独立的[八工具任务环境](wearable_tasks/README.md)和 2,900 条正式任务：训练 2,000、验证 300、三类测试各 200。参考执行、重放、隔离与隐私检查通过，并完成 12,968 次构造负例拒绝检查，报告见 `data/wearable_tasks/formal/validation-report.json`。这不是实际模型的任务成功率。

统一 SFT 数据含约 9,000 条训练、700 条验证和三类各 200 条测试记录，保留独立元数据及真实 Qwen tokenizer 审计。4096-token 视图整条跳过原有 3 条超长记录，实际为 8,997/700；383/23 条短样本只用于冒烟。见[数据准备结果](docs/sft-data-v1-results.md)及[重新处理说明](docs/sft-data-v1.md)。

固定 revision 的 Qwen3.5-2B 模型权重、tokenizer 与独立数据/训练环境均已准备。本机 P1000 已完成 20 步短样本冒烟，以及当前最长 3816-token 完整轨迹的受控前向、反向和一次临时更新；整轮完整 SFT 尚未执行。相关测试 49 项通过，见[长文本实测报告](reports/p1000-long-context-report.md)及[可选训练指令](docs/sft-training-v1.md)。

已新增 `wearable_assistant` 固定模型系统：正式八工具模型运行器、三组无训练对照、状态管理、写后核验、SQLite 会话恢复、可重放日志、600 条任务批量评测和本地 Web/API。代码验收与真实模型能力评测分开，真实推理、完整评测及真实业务连接留给本机手动运行。完整指令见[助手系统说明](docs/assistant-system.md)。

```powershell
# 无模型调用的配置/数据检查
& './scripts/run_assistant.ps1' -Task plan
# 启动界面/API（需先准备配置中的推理端点）
& './scripts/run_assistant.ps1' -Task serve -Execute
```

原 `lightagent` 环境和训练模块保留。SFT → GRPO 的训练桥接仍未完成；[旧就绪报告](reports/remote-experiment-readiness.json)是该训练流程的历史审查，不代表新增推理系统的当前状态。

训练入口使用 Hugging Face TRL，默认只输出计划，只有 `--execute` 才会训练。旧配置与早期交付记录保留供追溯，不应将其中的待执行状态或旧数据路径视为当前实验入口。

早期第 4–8 部分的接口与验收说明见[实施说明](docs/implementation-4-8.md)。下方快速运行命令主要针对原 `lightagent` 环境，不等同于当前八工具任务集的完整模型评测。

规则参考策略只支持本文的条件运动任务，用于检查环境和执行链路。它的成绩不代表 Qwen 的模型能力。模型端点接通后，才能测量模型的工具选择、参数生成和任务完成能力。

## 快速运行（Windows PowerShell）

请显式使用项目虚拟环境的 Python；原轻量运行环境为 `.venv`，数据与训练环境分别为 `.venv-data`、`.venv-train`。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 晴天、无冲突：创建跑步及提醒，并分别回读核验
.\.venv\Scripts\python.exe -m lightagent demo --trace --output outputs/demo.json

# 下雨：改为室内拉伸
.\.venv\Scripts\python.exe -m lightagent demo --case rain --trace

# 冲突：先询问；用户同意后顺延至 19:00，不覆盖已有会议
.\.venv\Scripts\python.exe -m lightagent demo --case busy_confirmed --trace

# 提醒创建失败：保留日程，明确报告部分完成
.\.venv\Scripts\python.exe -m lightagent demo --case reminder_failure --trace

# 完整固定回归评测与单元/集成测试
.\.venv\Scripts\python.exe -m lightagent eval --output outputs/eval-rule.json
.\.venv\Scripts\python.exe -m pytest -q
```

安装完成后也可以使用 `.\.venv\Scripts\lightagent.exe`。Linux/macOS 使用 `python3` 创建环境，并将示例解释器路径替换为 `.venv/bin/python`。

## 交互与演示

```powershell
.\.venv\Scripts\python.exe -m lightagent chat --trace --output outputs/chat.json
```

示例输入：

> 帮我检查明天下午六点是否有空。如果没有安排，而且杭州不下雨，就安排半小时户外跑步；否则安排室内拉伸，提前十分钟提醒我。

默认交互环境读取当前 `Asia/Shanghai` 时间，并为列出的八个城市、今天起四天配置模拟晴天。**这不是实时天气服务。** 规则基线缺少城市、时间、时长或提醒提前量时会询问，可继续回复“城市是杭州”等补全信息。时间有歧义时应补充完整表达，如“明天下午六点”。

`/state` 查看实际状态；`/new` 开始另一个任务并保留当前进程内已有日程；`/quit` 退出。JSON 输出保存轨迹用于检查，当前不支持从该文件恢复聊天或定时推送。

使用可重复的固定环境：

```powershell
.\.venv\Scripts\python.exe -m lightagent chat --scenario examples/hangzhou.json --trace
.\.venv\Scripts\python.exe -m lightagent cases
.\.venv\Scripts\python.exe -m lightagent tools
```

固定演示的当前时间为 **2026-11-16 10:00，Asia/Shanghai**。`busy` 只展示询问；`busy_confirmed` 自动输入预先编写的测试回复“同意顺延”。可用 `--reply 取消` 覆盖演示回复。演示、评测和真实交互均不向外部日历写入，提醒只是环境记录。

## 接入模型推理

项目使用 `/v1/chat/completions` 兼容协议，连接你已经部署的模型服务。该参数不限定服务提供商，也不会下载或启动模型。密钥只从环境变量读取，轨迹不保存鉴权头；轨迹会包含用户文本及工具观察。

```powershell
$env:MODEL_BASE_URL = "http://127.0.0.1:8000/v1"
$env:MODEL_NAME = "Qwen/Qwen3.5-2B"
# 仅当服务要求鉴权时设置：
# $env:MODEL_API_KEY = "..."

.\.venv\Scripts\python.exe -m lightagent demo --backend openai --label base --trace
.\.venv\Scripts\python.exe -m lightagent eval --backend openai --label base --output outputs/eval-base.json
```

服务需要支持原生 `tools`/`tool_calls`，每步一个工具调用，并在不调用工具时输出 `Final` JSON。系统提示已经描述格式。当前不兼容仅把工具调用打印成普通文本的端点；多工具并行调用会被明确拒绝。用 `--timeout` 控制单次网络超时，`--max-steps` 控制每轮最大动作数，避免无界循环。

`base`、`sft`、`rl` 仅为结果标签，未来可对不同模型检查点使用同一接口。**当前没有连接真实 Qwen 服务，也没有生成任何模型评测成绩。** 本机 Quadro P1000 为 4GB 显存，当前仅安装轻量原型依赖；训练硬件和模型服务另行准备。

## 工程结构

```text
src/lightagent/
  models.py                 参数、动作、结果和场景类型
  timeparse.py              有边界的中文时间解析，歧义/过去时间拒绝
  environment.py            十四个工具、状态变化、幂等、故障注入和日志
  policy.py                 规则参考策略、模型端点策略
  runtime.py                多步执行、澄清后继续、步数限制和反馈核验
  cases.py                  手写固定回归场景（不是训练集）
  evaluation.py             独立检查实际状态、约束与执行过程
  integrations/trl_env.py    TRL environment_factory 工具契约适配
  goals.py / user_simulator.py  私有目标谓词与受控用户反馈
  data_pipeline.py          xLAM/ToolACE 安全转换、清洗与分组划分
  training.py               TRL SFT/GRPO 计划、mask 审计与训练入口
  rewards.py / replay.py    有界奖励、组内诊断与轨迹复算
  benchmarks.py / profiling.py  BFCL 启动与流式 HTTP 效率采集
  cli.py                    demo / chat / eval / cases / tools
examples/hangzhou.json       可修改的本地环境
tests/                      时间、工具、执行循环、HTTP 和适配层检查
docs/architecture.md        行为约定及架构
docs/trl-roadmap.md         后续 TRL 接入边界
```

## 评测口径

20 个固定场景覆盖晴/雨/多云、已有安排、写入前临时冲突、确认/取消顺延、参数缺失及补全、创建失败、提醒失败、回读失败、天气缺失、服务不可用、无空档和时区跨日。

- `outcome_accuracy`：是否得到该场景应有的结果。正确澄清、正确报告失败或部分完成也计为正确。
- `task_completion_rate`：是否实际创建了正确日程及提醒，并返回核验后的成功。失败或澄清不计入完成。
- 工具调用数、工具错误数、耗时、失败类型：用于检查效率和定位问题。故障注入产生的预期错误不等于评测失败。

评测不依赖最终回复中的一句“已完成”，而是检查活动分支、城市、时段、时长、日程冲突、提醒关联及提前量，以及创建前查询和顺延许可。当前是小规模工程回归集，不能作为外部泛化 Benchmark 或简历中的模型成绩。

## 训练入口与后续范围

当前机器只运行轻量验证。以下命令不读取训练数据、不创建 Trainer：

```powershell
.\.venv\Scripts\python.exe -m lightagent.training --config configs/sft.json
.\.venv\Scripts\python.exe -m lightagent.training --config configs/grpo.json
.\.venv\Scripts\python.exe -m lightagent.benchmarks --config configs/bfcl-smoke.json
```

`--execute` 才会实际训练或运行公共基准。当前不运行它们。SFT 需要用户准备的公开数据文件与本地模型；GRPO 需要外部明确提供的可执行任务，项目不会为此自动生成自建数据。公开数据的历史工具返回仅供 SFT，不能充当 RL 环境。

1. 接通原始 Qwen3.5-2B，记录无训练基线，分析失败分布。
2. 准备公开示范后，使用已提供的清洗管线、真实 mask 审计和 TRL `SFTTrainer` + LoRA 入口。
3. 获得可执行任务与训练 GPU 后，实测 TRL `GRPOTrainer`、奖励消融和组内学习信号。
4. V1 完成后再扩展自动 Rubric，并校准其与确定性校验的差异；后续再考虑图片输入。

当前验证覆盖普通 Python 接口、工具状态、HTTP 协议、数据清洗、mask 逻辑和奖励漏洞；**尚未使用真实 Qwen tokenizer/权重运行 TRL Trainer**，不能据此宣称训练兼容性或能力提升。详见 [TRL 路线](docs/trl-roadmap.md)。
