# CPU 可执行穿戴助手任务工厂

**当前交付状态：代码已写入，按用户要求没有执行。** 没有运行 unittest、生成 100/2,900 条任务、更新实测报告或运行训练。当前状态见 `reports/wearable_tasks_pending.json`；下文的数据数量是运行目标，不是已经生成或通过的数量。

## 范围与复用

沿用项目现有工具的半开区间、原子修改、提醒联动和状态隔离语义。现有 `lightagent` 依赖 Pydantic、使用 14 个工具及另一种返回/奖励契约；为冻结附件要求的八工具接口，新增本目录的标准库实现，不修改旧工具、公开 Hermes/ToolACE 数据或已有训练入口。

- Python >=3.10，仅标准库；不需要 pip 安装本模块。
- 单进程、单线程、CPU、内存状态和 JSON/JSONL。
- 不导入 torch、transformers、vLLM；不访问网络、外部模型、天气、日程或账户。
- 固定模拟时钟与 +08:00，不使用机器当前时间生成答案，不支持 DST、多时区旅行、重复日程。
- 无真实通知和医疗/运动处方；活动只是日程目录标签。
- 训练框架方向仍是 TRL；本次不下载模型、不运行 SFT/GRPO。

## 用户本地运行指令

从项目根目录运行。**以下命令由你执行，本次交付时未执行。** 默认 `python` 可能是 3.7，因此使用 `py -3.10` 或现有虚拟环境解释器。

推荐使用顺序脚本；任一步退出码非零就停止：

```powershell
Set-Location -LiteralPath 'D:\ms-swift＋Qwen3.5-2B'

# 先分阶段运行 unittest，再生成、执行与验证 100 条冒烟任务
powershell -NoProfile -File .\scripts\run_wearable_tasks.ps1 -Stage smoke

# 只有前一步完整通过后，才生成与验证正式 2,900 条任务
powershell -NoProfile -File .\scripts\run_wearable_tasks.ps1 -Stage formal
```

也可用 `-Stage all` 连续执行；重新生成到已有目录需显式 `-Overwrite`。只覆盖本模块固定的已知文件，不递归删除目录，不覆盖其他项目文件。脚本优先使用系统 `py -3.10`，不存在时使用项目解释器，并加 `-E -S -X utf8`：忽略 Python 环境变量、跳过 site-packages 和 .pth、固定 UTF-8。本模块只依赖标准库，因此无需加载现有虚拟环境中的包。脚本采用 ASCII 字符，兼容 Windows PowerShell 5.1。不修改执行策略；若脚本被系统策略阻止，可以逐条运行下面的 Python 指令。

分步指令（前一步失败就停止）：

```powershell
py -3.10 -E -S -X utf8 -m wearable_tasks check --output outputs/wearable_tasks/unit-tests.json

py -3.10 -E -S -X utf8 -m wearable_tasks build --config wearable_tasks/configs/smoke.json --output data/wearable_tasks/smoke --checks-report outputs/wearable_tasks/unit-tests.json

py -3.10 -E -S -X utf8 -m wearable_tasks build --config wearable_tasks/configs/formal.json --output data/wearable_tasks/formal --checks-report outputs/wearable_tasks/unit-tests.json --smoke-report data/wearable_tasks/smoke/validation-report.json
```

默认种子为 42，可在 build 命令使用 `--seed`，或脚本使用 `-Seed`，但冒烟与正式种子必须相同。代码或工具契约改变后必须重新检查和生成冒烟报告。

仅查看冻结 Schema（不会生成数据）：

```powershell
py -3.10 -E -S -X utf8 -m wearable_tasks schema
```

正式生成成功后，选择一个已有任务重放：

```powershell
$taskRef = Get-Content -LiteralPath 'data/wearable_tasks/formal/public/grpo/test_composition.jsonl' -TotalCount 1 | ConvertFrom-Json
py -3.10 -E -S -X utf8 -m wearable_tasks replay --directory data/wearable_tasks/formal --case-id $taskRef.case_id
```

## 输出与预期规模

| 配置 | train | val | test_id | test_composition | test_robustness | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 40 | 12 | 16 | 16 | 16 | 100 |
| formal | 2,000 | 300 | 200 | 200 | 200 | 2,900 |

每组完整保留四个变体，因此每个 split 是四的倍数。冒烟只验证管线，不作为正式训练样本追加到正式数据；两种 profile 不应合并，以免重复基础场景。

运行成功后分别在 `data/wearable_tasks/smoke/` 和 `data/wearable_tasks/formal/` 产生：

```text
validation-report.json            实际数量、耗时、调用分布、失败与校验结果
private/
  train.jsonl, val.jsonl, test_*.jsonl  完整私有任务
  group-plan.json                 生成变体之前的语义分组与 split 计划
  negative-examples.json          各类实际被拒绝错误样本的诊断摘要
  trajectories/*.jsonl            全 split 可重放参考轨迹与私有判定信息
public/
  tool-schema.json                冻结工具版本、Schema hash 与定义
  sft/train.jsonl, sft/val.jsonl   执行成功的公开对话；不导出测试示范
  grpo/*.jsonl                    case_id 加公开初始消息和 Schema
```

中途失败会写 `status=failed` 报告，并保留 `.jsonl.part`，不会将未通过的数据发布为本次成功产物。加载器拒绝读取未通过完整验证的私有库。重用目录时旧文件可能仍在，必须以本次 `validation-report.json` 为准；默认不覆盖非空目录。

## 冻结的八个业务工具

契约版本 `wearable-tools-v1`，输入都必须是 JSON 对象，拒绝多余字段、缺参、布尔冒充整数、无时区或非 +08:00 时间。

| 名称 | 语义 |
| --- | --- |
| list_events | 返回与 `[start_time,end_time)` 相交的事件，按时间和 ID 排序 |
| create_event | activity_id、start_time、duration_minutes；只检查并执行，不规划备选 |
| update_event | 原子修改 activity_id/start_time/duration_minutes；排除自身冲突，同步提醒 |
| delete_event | 删除指定对象和关联提醒；不存在则 NOT_FOUND |
| get_weather | 从固定表返回 city/date/condition，不给行动建议 |
| search_activity | 按 ID、室内属性或目录时长返回候选事实 |
| set_reminder | 每事件唯一提醒，重复同参数幂等；正整数提前量，触发时间不早于模拟现在 |
| get_preferences | 返回当前任务可查询的既有偏好 |

所有事件 ID 单调分配，不复用已删除 ID。写入失败不分配 ID、不产生部分写入；相邻事件首尾相接不冲突。修改活动 ID 时标题同步为目录标题；仅改时间或时长时其他字段不变。

统一返回 `{"status":"success","data":...}` 或 `{"status":"error","error_code":...,"data":...}`。包含 INVALID_ARGUMENT、NOT_FOUND、TIME_CONFLICT、TEMPORARY_UNAVAILABLE；超预算额外返回 BUDGET_EXCEEDED。

`ask_user` 与 `final` 是环境内部动作，不计为业务工具。ask_user 只回答明确请求且尚缺失的槽位，未询问信息不自动提供。L6 用户改意图只在 create_event 成功后触发，并必须在最终结束前处理。

## 任务与参考策略

先构造结构化语义组，再指定 split，然后生成初始世界、结构化用户意图、中文请求、目标和四个组内变体。语义组由任务种类及事件数量、间距、持续时间、目标位置与提醒关系决定，不用姓名/城市/日期直接定义新组。每组至少有两个相似事件，间距是实际关系而非闲置标签。

- L1：天气、时段日程、单事件创建。
- L2：先查对象再提醒/修改/删除；创建后提醒。
- L3：天气分支；查询偏好与目录后选择符合约束的最短活动。
- L4：先询问缺失的开始时间或时长。
- L5：只读首次临时失败重试；遇到冲突后按用户授权的有限备选顺序尝试。
- L6：创建后改时或撤销；组合测试保留“天气→创建→用户改时→提醒”。

复述、日期/实体变化、晴雨反事实均留在同组。天气条件任务按相同请求配成晴雨两支，正确活动随实际工具返回改变。模拟当前时间明确写入可见请求，避免不同基础世界发生完全相同的跨 split 请求；这不能代替结构分组，所以同时检查语义组、去元数据后的完整任务和请求重复。

参考策略只持有内部结构化用户意图，以避免实现中文解析器；它只通过 `step` 查询和修改，天气与偏好从工具获取，缺参从用户状态机获取，后续新意图从真实触发回复获取。策略不读取 goal、初始状态、故障计划或隐藏答案。测试中使用仅暴露 step 的代理约束这一点。

## 判定与防投机

只用终局二值奖励：全部必要条件满足为 1，否则为 0。没有过程奖励、Judge 或自由文本关键词评分。

判定覆盖最终事件与提醒的精确集合、非目标对象/字段保留、唯一提醒与触发时间、结构化查询事实、必要查询在决策写入前发生、缺参在写入前真正询问、冲突由真实错误发现、L6 最后指令得到处理、预算和正常终止。只对有语义需要的任务要求查询证据；普通明确创建不强制先查日程，不要求复刻全部参考步骤。

除了最终状态，还检查成功写入是否由用户授权：错建再修正、误删其他对象后补建、额外创建后撤销等不能靠恢复终态绕过。查询结果来自内部 final.result，与实际查询返回作结构化比较。自然语言回复的完整语义忠实度尚未自动验证。

全量验证时逐任务执行参考策略、同动作重放、四份环境隔离和公开信息检查；对适用任务批量注入错误活动/时间、漏提醒、绑错对象、重复创建、误删非目标、猜中缺参、事后补问、忽略改意图、跳过必要查询及伪造查询结果。任何错误样本被接受都会让构建失败。

## 训练接入边界

SFT 输出仅包含 `messages` 与 `tools`。原生 assistant.tool_calls、tool_call_id、真实工具返回、澄清和后续用户消息全部保留；不带 goal、参考意图、评分、测试标签或长篇思维链。测试集示范只保留在私有评测轨迹中。

GRPO 输出是环境加载引用，不预生成模型候选。`ReferenceEnvironmentLoader.reset(case_id)` 返回白名单 observation；`step(action)` 返回本步可见消息、done 与奖励；`terminal_reward()` 只在结束后可取。

**GRPO 文件不是可直接交给旧 TRL Trainer 的即用格式。** 当前新增的纯推理接入位于 `src/wearable_assistant/`，支持 case_id 私有加载、原生工具调用、有限澄清协议、多轮用户触发、执行日志及批量评测，使用方式见[助手系统](../docs/assistant-system.md)。统一 SFT 数据处理及 mask 审计已另行实现，见[数据准备结果](../docs/sft-data-v1-results.md)。GRPO 训练桥接仍未接通；纯推理系统不会训练参数，也未修改本目录冻结的环境/判定器 Python 源码。

## 文件与验收报告

模块包括 `contract.py`、`environment.py`、`templates.py`、`reference.py`、`validation.py`、`auditing.py`、`exporting.py`、`pipeline.py`、分阶段 `checks.py` 与 unittest。没有自动运行入口、后台进程或导入时生成数据行为。

运行后的报告统计来自实际执行：Python、种子、数据/契约/源代码 hash、各 split/等级/模板/分支数量、参考成功与失败原因、错误样本拒绝数量、隔离/重放/泄漏/分组结果、调用次数分布、实际耗时和每个已发布 JSONL 的 SHA256。内存默认标“未测量”，不猜测峰值。

早期交付时的未执行状态仅供历史追溯。当前正式数据已构建并通过验收，见 `data/wearable_tasks/formal/validation-report.json`；纯推理接入的工程验收另见 `reports/assistant-system-implementation.json`。参考执行或单元测试不等同于实际模型的 600 条测试成绩，后者仍需本机手动运行。
