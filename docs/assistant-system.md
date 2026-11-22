# 固定模型八工具助手：运行与验收

本模块不训练模型，也不在导入、预览或测试时下载权重。模型保持固定，通过工具执行、公开会话状态、执行校验和离线评测组成闭环。当前提供本地模拟环境和 HTTP 业务网关适配器；没有连接任何真实用户的天气、日历或提醒账号。

## 1. 已实现的模块与边界

| 模块 | 实现 |
| --- | --- |
| 模型接口 | 兼容 Chat Completions 的 HTTP 客户端；八个原生业务工具；每轮单调用，多个调用整体拒绝 |
| 原生协议 | arguments 对象/JSON 字符串；严格 JSON、重复键/NaN 拒绝；固定 Qwen XML 工具协议；澄清与最终回复 |
| 会话 | 公开槽位、意图版本、已观察对象、待处理调用、失败状态、SQLite 持久化、重启恢复 |
| 执行 | 参数检查、可见请求授权规则、对象证据、重复写入拦截、只读临时失败一次重试、事件写后回读、提醒回执时间校验 |
| 取消 | API 可在模型调用期间登记停止请求；在途调用返回后停止后续步骤；不自动撤销已有写入 |
| 评测 | 三类各 200 条，支持单组/三组对照、逐案例落盘、配置/数据/代码签名、断点续跑和失败统计 |
| 日志 | 模型原始响应、控制器拦截、工具回执、用户后续消息、耗时和 usage；原始成功声称与最终受控结果分开统计 |
| 产品入口 | 终端对话、本地 Web 和 HTTP API；Web 文本通过 textContent 显示 |
| 推理服务 | 可选 Transformers 单 GPU 本地端点，使用已存在权重；不更新参数 |

控制器只持有 `backend.observation()` 和实际返回。私有 goal、reference_intent、fault_plan、未询问答案不进入模型消息或公开会话状态。`TaskBackend.offline_evaluate()` 仅在模型执行停止后由评测器调用。

授权、时间/时长和自然语言澄清识别是有界规则，不能保证任意中文表达的完整语义正确。结构化澄清支持 `start_time/end_time/duration_minutes/city/date/activity_id/event_id/minutes_before`，以及部分常见自然问句；正式任务的用户模拟器仍只回答该任务实际缺失的槽位。建议回答“明天18:00”“持续30分钟”、`city=杭州` 或 `start_time=2026-11-19T18:00:00+08:00`。不适用任务的澄清会被环境拒绝。自然语言回复的完整忠实度没有 Judge 保证。

最终 success 表示控制器可见证据检查通过，不等同于已经通过私有目标评测。正式能力结论以离线测试报告为准，特别关注 `blocked_valid_completions`（控制器误拦截）、`false_success_claims`（模型原始虚报）与 `reported_false_successes`（最终对外虚报）。

工具返回中的 `status=success` 只描述一次工具执行；整项交互看 `state.phase` 和 `final.status`。例如天气已查询成功，但模型反复查询直到预算耗尽，整项任务仍是 failed。界面会显示真正的停止原因和累计 token 用量，保留工具原始日志。

当前 full 组增加以下控制，baseline/state 组保持原生调用循环：

- `direct_weather_completion=true`：仅对单一用户消息中明确的单城市、单日期天气查询，在实际成功回执与请求参数完全一致后，由控制器直接生成天气答复。支持“查询杭州明天的天气”等有界表达；多日期、多城市、穿衣建议、天气后安排日程等不走此路径。用户历史已有写入或待处理改意图时也不走此路径。标记 `final_origin=controller_weather`，`raw_final=null`，不可宣称为模型自主完成；正式报告单列 `controller_weather_completions`，仍以离线目标评测决定是否成功。
- `reuse_successful_reads=true`：本地/正式模拟环境内，相同意图下成功的相同只读查询可复用真实回执，返回 ALREADY_OBSERVED 提示并记录原调用 ID；失败不缓存。任何写入尝试或新用户消息都会失效缓存。真实 HTTP 工具不复用缓存，避免把外部已变化状态当作当前结果。
- `max_repeated_read_attempts=3`：同一意图且没有写入尝试的间隔内，相同只读请求最多处理三次，第三次处理后立即停止，不再请求第四轮模型。成功回执仍可在明确的简单天气查询中直接答复；其余重复读取保留协议配对，并说明尚未完成全部任务。
- `max_consecutive_tool_failures=3`：同一任务连续三次工具调用失败，第三次返回后立即结束。实际工具错误和控制器拦截都计数，临时失败自动重试也计数；改变日期、城市、工具或错误码不能重置计数。成功调用或新的用户输入重置连续失败计数。结束时 `final.result.error_code=CONSECUTIVE_TOOL_FAILURE_LIMIT`，保留最近三次错误；有 NOT_FOUND 时明确说明未找到数据。若此前已经成功写入，则返回 partial 并披露已生效对象，不声称全部失败或自动回滚。计数和错误历史随快照保存；评测单列 `consecutive_failure_stops`。
- `grounded_assistance=true`：创建/变更活动前，通过实际 `search_activity({})` 回执读取目录。仅当用户明确的活动名称或 ID 与目录唯一匹配时，才可修正模型编造的活动 ID；时间、时长不改，原模型调用和 `argument_repair` 都保留。无唯一匹配不猜测。完整目录查询同样计入工具预算；不会直接读取模拟世界中的隐藏状态。
- 简单的“查看我的运动偏好”“查询明天的日程”在参数和成功回执匹配后可直接答复；单次明确的“明天18:00安排30分钟户外跑步”只有写入回执与参数一致且回读核验通过后才可收尾。分别标记 `controller_query` 和 `controller_verified_write`，评测单列计数，不当作模型自主完成。多城市、多日期、创建加提醒等复合请求不走这些快速收尾路径。

例如“查询南京明天天气”依次产生南京 2026-11-19 的 NOT_FOUND、2026-11-18 和 2026-11-17 的 DATE_NOT_PROVIDED_BY_USER，会在第三次错误后返回失败，说明未找到南京 2026-11-19 的天气及日期不符合请求；不会继续倒推其他日期。这里停止的是当前任务，Web/API 和模型服务继续运行。

更新后重启 Web/API 服务，访问 `/health` 确认 `controller_revision` 为 `verified-completion-v7`。已有会话可继续使用，旧活动和提醒记录在恢复时根据成功回执与本地模拟状态重建。

本地 `full` 会话中可以先新建“明天18:00安排30分钟户外跑步。”，然后在同一会话依次继续“给刚创建的活动设置提前10分钟的提醒。”“把刚创建的活动改到20:00，时长和活动不变。”“核对关联提醒。”。明确的单操作续接由控制器使用同一会话的对象 ID 和已提供参数执行。多个活动无法唯一匹配时先询问 ID，新会话中的“刚创建”也会先澄清。改期回读检查活动数量、原 ID、保留字段和关联提醒 ID、时间偏移；有活动回执或回读证据但提醒验证失败时返回 `partial` 并说明具体未核验内容。提醒回读在本地模拟控制器内完成，模型仍使用原八个工具。

长对话中的约束和澄清记录保存在 SQLite。可输入“这周只安排室内活动。”或“以后只安排室内活动。”，控制器按活动日期和本地活动目录校验写入。用“取消本周室内活动限制。”撤销本周规则，或用“取消室内活动限制。”撤销两个范围的室内规则。单个活动可用“户外跑步后续只改开始时间。”保护时长和类型，用“取消户外跑步的只改开始时间限制。”撤销。支持用名称与时间组合指定活动，例如“把明天18:00的户外跑步改到明天22:00。”。同名活动会列出候选及时间，选定 ID 后继续。待补参数可以隔着普通对话继续回答，输入“取消刚才的请求。”可取消挂起的请求。这些明确表达使用有限规则识别，其他表达沿用模型工具调用。

本地 `full` 组对明确的创建、改期、取消与提醒请求核对本轮写入和相应核验标记，最终对象 ID 必须对应本轮证据。显式组合请求需要完成识别出的全部写入步骤。`local_fast_path=true` 时，单日查询和目录名称、开始时间、时长都明确的单次创建可直接调用工具。创建仍经过目录匹配、写入校验和回读。界面和轨迹用 `controller_fast_path` 标记此来源。

工程回归使用脚本模型与模拟工具，只证明对应控制逻辑成立，不证明真实基座模型能完成全部自然语言请求。2026-11-18 的真实日志出现模型将“户外跑步”编成 `running_outdoor` 而实际目录 ID 为 `run`；这是目录依据不足和模型参数生成错误。三次停止只能阻断循环，不能提升任务成功率。目录解析与有限直接收尾是公开标记的系统辅助，复杂修改、取消、组合提醒等仍须独立模型实测，不能据单元测试宣布可用。

这些是系统控制策略，评测清单会保存配置及代码签名；不能把新旧策略报告混作同一个实验。若要单独衡量“模型自行读取反馈并结束”的能力，将 `direct_weather_completion` 设为 false。所有策略保留完整工具协议和执行日志，不删除 Schema、错误返回、澄清或用户改意图消息来缩短上下文。

## 2. 本机准备与无推理检查

在项目根目录的 PowerShell 运行。轻量 API/控制器使用现有 `.venv-data`；仅本地模型端点使用 `.venv-train`。也可使用独立 Python 3.11 环境安装本项目。不要把原虚拟环境直接拷贝到 Linux。

```powershell
Set-Location -LiteralPath 'D:\ms-swift＋Qwen3.5-2B'
$ErrorActionPreference = 'Stop'

# 仅安装本地项目及开发依赖，不下载模型；更新控制台入口和打包映射
& '.\.venv-data\Scripts\python.exe' -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw '项目安装失败' }

# 仅检查配置、固定数据哈希与 600 条任务的可用性，不请求模型
& './scripts/run_assistant.ps1' -Task plan

# 脚本模型与本地假 HTTP 端点的工程测试，不是模型能力测试
& './scripts/run_assistant.ps1' -Task check
```

`configs/assistant.json` 指定模型地址、名称和预算。默认模型名为 `Qwen/Qwen3.5-2B`，服务地址为 `http://127.0.0.1:8000/v1`。接入已有推理 API 时，只修改这里的 `endpoint.base_url`/`model`，并在 shell 设置 `$env:MODEL_API_KEY`。项目不自动读取 `.env`。同一轮对照不要更换端点背后的模型权重。

## 3. 手动启动固定模型推理（窗口一）

如果已有兼容端点，跳过此步骤。下面命令只有加 `-Execute` 才加载权重、占用 GPU；本次开发没有执行。端点只支持非流式推理，温度/seed 和完整 token 用量会记录；TTFT 保持 null。

```powershell
Set-Location -LiteralPath 'D:\ms-swift＋Qwen3.5-2B'

# 3090/A100 等支持原生 BF16 的设备：只做推理，不做 SFT/GRPO
& './scripts/run_assistant.ps1' -Task model -Execute
```

本地端点使用 `.venv-train` 中已安装的 Transformers/torch/accelerate；模型文件来自 `models/Qwen3.5-2B`，`local_files_only=True`，不会自动下载。工具标签使用当前模型模板的 XML 格式解析，返回标准 tool_calls 给控制器。输入加输出超过 8192 时拒绝，不截断工具协议。

P1000 可单独选择以下实验性模式，不要同时启动两份模型服务：

```powershell
& './scripts/run_assistant.ps1' -Task model -Pascal -Execute
```

Pascal 模式为 NF4/FP32、eager、4096 上限；本推理服务尚未在真实 GPU 上实测，不由之前训练试跑推断其速度或完整覆盖率。长对话可能超长，不能删 Schema/反馈强行运行。已提供 `configs/assistant-p1000.json`，将每次响应上限设为 256 token、单请求超时提高至 600 秒、本轮时间上限设为 1200 秒；在窗口二使用 `-Config configs/assistant-p1000.json`。这些是待本机验证的保守预算，不保证所有长任务都能装入 4096 上下文，P1000 应先做短任务冒烟。

P1000 的上下文窗口是 4096 token，预留 256 个生成 token 后，模型输入上限是 3840。完整对话、状态与执行轨迹保存在 SQLite。本地 `state` 和 `full` 组用本地 Qwen tokenizer 和同一对话模板计算工具 Schema 与消息的实际长度，将当前一轮、至多两轮近期对话与相关对象状态发给模型。超限时先移除较早轮次，若当前轮仍超限，对日程和活动目录列表生成带总数、保留数和省略数的模型输入视图，优先保留当前请求涉及的记录，工具调用与返回保持配对。SQLite 中仍可查看原始结果。必要内容超限时提示缩短请求。启用此功能的 Web/API 环境需要 `transformers` 和本地 tokenizer 文件。本地交互组按轮应用 `runtime.max_total_tokens`；原冻结任务评测仍按完整任务应用该预算。

更新控制器后，只需在 Web/API 所在窗口 Ctrl+C，再执行下面命令，然后刷新页面并新建任务；模型端口 8000 可保持运行，旧会话的失败结果保留原样：

```powershell
Set-Location -LiteralPath 'D:\ms-swift＋Qwen3.5-2B'
& './scripts/run_assistant.ps1' -Task serve -Config 'configs/assistant-p1000.json' -Execute
```

## 4. Web/API 或终端（窗口二）

```powershell
Set-Location -LiteralPath 'D:\ms-swift＋Qwen3.5-2B'
& './scripts/run_assistant.ps1' -Task serve -Execute
```

浏览器打开 `http://127.0.0.1:8765`。先试“查询杭州明天的天气”，再试“明天18:00安排户外跑步，时长请先问我”，回答“持续30分钟”，随后说“请把刚创建活动改到2026-11-19T20:00:00+08:00”。检查时间线与真实对象回执。

默认 `examples/assistant-world.json` 使用固定模拟时钟 `2026-11-18T09:00:00+08:00`，不会跟随机器日期。新会话加载该世界，已有会话重启时从日志重建自己的状态。不同会话世界隔离。

也可只使用终端：

```powershell
& '.\.venv-data\Scripts\python.exe' -B -X utf8 -m wearable_assistant chat --database outputs/assistant/chat.sqlite3 --execute
# 之后恢复：追加 --session 上一次输出的 session_id
```

`/cancel` 只停止后续操作；需要撤销已经创建的对象时，明确发出“撤销刚创建的活动”，由模型调用 delete_event。SQLite 保存完整会话以便恢复，**一个数据库只由一个服务/CLI 进程负责执行**，当前没有分布式执行租约。不同进程使用不同数据库。

API：

| 请求 | 用途 |
| --- | --- |
| `GET /health` | 服务和模型配置，不加载模型 |
| `POST /sessions` | `{request, request_id, profile?, defer?}` 创建任务；defer=true 只创建不调用模型 |
| `POST /sessions/{id}/run` | `{request_id}` 执行已创建任务 |
| `GET /sessions/{id}` | 公开会话状态和时间线，不含私有目标或后台世界 |
| `GET /sessions` | 最近会话 ID 和阶段，供恢复页面使用；不重新提交模型请求 |
| `POST /sessions/{id}/messages` | `{text, request_id}` 继续任务 |
| `POST /sessions/{id}/cancel` | `{request_id}` 在途调用返回后停止后续步骤 |

每个写 API 请求需要唯一 `request_id`；网络失败时原样重用同一 ID，同 ID 换内容会被拒绝。Web 先创建 deferred 会话获取 ID，再执行，运行中可以登记取消；失败后可刷新、重试未确认请求。进程中断时若工具尚无回执，恢复不会自动重发写操作；HTTP 网关超时也标记 WRITE_OUTCOME_UNKNOWN，需要检查外部实际状态。

页面每两秒读取持久化状态，并显示等待秒数、模型请求与工具调用次数。模型请求发出前保存 waiting_for_model 阶段；进度查询不获取执行锁、不重建环境，也不会发起额外推理。当前仍为非流式生成，等待秒数不代表已生成 token 数或完成百分比。刷新页面可恢复已保存的会话 ID，也可点击“恢复最近会话”。

如果一直显示 ready，或者日志出现 `Model response was truncated`，请检查是否仍在运行旧服务。旧推理端点沿用模型配置的 endoftext=248044，而当前 tokenizer 的聊天结束标记 im_end=248046，可能在回合结束后继续生成到输出上限。修复版显式传入两个停止 ID，启动行应包含 `EOS=[248046, 248044]`。不需要改模型配置或 tokenizer 文件。

更新代码后，在两个原 PowerShell 窗口分别 Ctrl+C，然后重新运行对应 model/serve 命令，浏览器 Ctrl+F5 强制刷新。原任务如果已确定失败，用新建任务再次验证；“重试未确认请求”保留原 request_id，不会把已失败任务当新任务重跑。本次修复没有自动重启用户进程，也没有额外发起真实生成请求。

服务默认只监听本机。非回环监听需要 `ASSISTANT_API_TOKEN`，配合实际部署的 TLS 反向代理；浏览器令牌仅存页面内存。API 是轻量开发服务，不包含多租户身份体系、生产限流或独立账号权限管理。不要把未经配置的本地开发服务当作公网生产系统。

## 5. 手动运行评测：先验证集，再冻结测试

以下步骤会真正请求模型，均留给本机执行；本次没有运行。每个目录拒绝覆盖；继续原任务必须显式 `-Resume` 并使用相同模型、代码、数据和配置。中断发生在案例中途时，该案例在全新的模拟世界重新开始，不重放真实外部写操作。

```powershell
# 先用验证集前 5 条检查端点和工具协议；不能当正式结果
& './scripts/run_assistant.ps1' -Task smoke -Output outputs/assistant/val-smoke -Execute

# 正式完整系统：三类测试各 200，共 600 条
& './scripts/run_assistant.ps1' -Task evaluate -Output outputs/assistant/full-test -Execute

# 中断后继续同一轮
& './scripts/run_assistant.ps1' -Task evaluate -Output outputs/assistant/full-test -Resume -Execute

# 三组分别运行同一套 600 条，共 1800 个模型任务，不进行训练
& './scripts/run_assistant.ps1' -Task compare -Output outputs/assistant/compare-test -Execute
```

三组基线使用同一核心提示、工具 Schema、温度、seed 和调用/时间预算：baseline 原生单工具循环；state 额外向模型提供公开会话状态；full 再加入执行校验、有限修复/重试和写后核验。各组独立世界，工具重试和控制器回读都计入工具预算。增加的状态会占用上下文，报告实际 token 开销，不宣称三组 token 使用量相等。

提示词和规则调试用 train/val；三个测试集保持冻结。后端读取报告与 JSONL 的 SHA256 并检查冻结环境源码指纹，防止误用已经变动的数据/判定器。服务端若不支持 seed 或返回不完整 usage，应先做端点兼容校验，不能把缺失 token 统计当零。

每组输出：

```text
manifest.json       配置、代码、任务资产哈希和签名
cases/{id}.json     每条模型轨迹、动作回执、原始/受控最终回复、离线判定
report.json         总体及各 split 的成功率、失败原因、澄清、改意图、虚报、成本和耗时
comparison.json     仅三组对照根目录生成
```

报告区分原始模型声称、控制器对外结果和独立终局判定；额外统计误拦截、重复创建、授权写入违规、协议错误、重试、恢复成功及 P50/P95。模型结果必须来自上述实际执行，单元测试中的脚本模型不填入正式能力报告。

交互式评测固定案例在 `examples/assistant-interactive-cases.json`，复用网页的 `SessionManager.create/run/message`，每组写入独立 SQLite。`baseline` 为原生调用，`memory` 加入本地任务记忆和裁剪，`full` 加上执行核验，`fast` 在 `full` 上开启明确请求快速路径。独立评分核对当轮写入、工具结果和后端状态，分别统计完成、正确澄清、正确拦截、错误写入、token 和耗时。误报分母为成功声称次数；误拦截分母为独立证据已满足的应完成轮次，计入其中拦住正确最终答复的情况。分母为零时报告 null。测试期望保留在评分器里。

下面启动一次有界真机对照，模型端口须已启动。省略 `--case` 可运行整套交互案例，其中包括插入 10 和 20 轮对话的长任务；`--output` 每次换用新目录。

```powershell
& '.\.venv-data\Scripts\python.exe' -B -X utf8 -m wearable_assistant --config 'configs/assistant-p1000.json' interactive --case simple-list --variants baseline memory full fast --max-model-calls 2 --output 'outputs/assistant/interactive-smoke-01' --execute
```

不调用模型的重放：

```powershell
& '.\.venv-data\Scripts\python.exe' -B -X utf8 -m wearable_assistant replay --artifact outputs/assistant/full-test/cases/替换为case_id.json
```

重放会对比真实环境返回与保存回执，重新计算离线判定。HTTP 外部写操作禁止重放，仅查看原回执；不会为了核验而再次执行真实写入。

## 6. 后续连接真实业务

实现了 `HTTPBackend`，接口为 `POST {base_url}/tools/{工具名}`，请求为冻结 Schema 的参数对象，响应为：

```json
{"status":"success","data":{}}
```

或 `{"status":"error","error_code":"TEMPORARY_UNAVAILABLE","data":{}}`。具体字段须与八工具本地契约一致，不能用空 data 代替本应返回的事件/天气事实。网关负责把厂商天气/日历/提醒 API 映射为该契约，并保证 Idempotency-Key 去重；本项目不包含任何厂商账号、密钥或已完成的实盘验证。

复制 assistant.json 新建独立配置，将 tool_backend 设为：

```json
{"kind":"http","base_url":"https://替换为你的业务网关","api_key_env":"TOOL_API_KEY","timeout":15}
```

在 shell 设置 TOOL_API_KEY 后显式启动对应配置。不要把密钥写入配置或模型消息。HTTP 后端默认使用新会话创建时的 Asia/Shanghai 当前时间，并冻结进会话快照；也可显式设置 now 用于网关沙箱测试，不要长期使用过期固定日期。HTTP 写请求使用会话/调用 ID 生成稳定幂等键；超时或无效回执不会自动重试写入。事件创建/修改/删除使用 list_events 回读，提醒只能验证回执与事件时间的一致性，八工具没有单独读取提醒接口，因此不声称完成了真实提醒投递验证。

## 7. 训练与实际结果

此系统运行无需 SFT/GRPO。原训练代码保留，手动训练说明见 [sft-training-v1.md](sft-training-v1.md)。旧 GRPO 的八工具训练桥接仍未实现，不属于本次推理系统的必要条件，也没有把其 20 步配置包装成正式训练。

本次只执行工程回归、假 HTTP 服务和 CPU tokenizer/模板检查，没有启动本地真实模型服务、真实业务连接、600/1800 条模型评测或任何训练。工程验收记录见 `reports/assistant-system-implementation.json`；实际能力结果留待本机运行后生成。
