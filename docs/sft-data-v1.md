# 统一 SFT 数据准备：手动执行说明

2026-11-16 在用户授权后已执行环境准备、tokenizer 下载、数据转换、自建重放、长度统计、混合与 mask 审计，未启动训练。[执行结果](sft-data-v1-results.md)列出真实数量和验证范围。下面保留重新准备时的分阶段指令；已有输出不会自动覆盖。旧 `lightagent.data_pipeline` 和旧训练配置保留，新入口为 `lightagent.sft_data`。

## 先准备环境与本地 tokenizer

从项目根目录执行，使用已有的 **Python >=3.11 训练环境**。该环境需要项目本身的依赖，以及 `requirements/train.txt` 中匹配的 Transformers / TRL；当前项目约定 `trl==1.18.0`。如需安装项目，可由使用者执行 `python -m pip install -e ".[dev]"`。不要使用 `wearable_tasks` 工厂专用的 `-S` 参数，因为这里需要第三方依赖。

本地 Qwen3.5-2B 目录必须包含真实 tokenizer、chat template 及必要配置。`AutoTokenizer` 始终使用 `local_files_only=True`，不会下载 tokenizer 或加载模型权重。数据准备和 mask 检查不需要 P1000、CUDA 或 bitsandbytes；它们会使用当前项目训练入口的 tokenizer/template 加载契约。

例如，把下面的解释器路径改为实际训练环境路径：

```powershell
Set-Location -LiteralPath 'D:\ms-swift＋Qwen3.5-2B'
$DataPython = '.\.venv-train\Scripts\python.exe'
$QwenTokenizer = 'models/Qwen3.5-2B'
```

本次实际环境为 `.venv-data`，本地 tokenizer 已保存在 `models/Qwen3.5-2B`；没有模型权重。下载来源、固定 revision 和上游哈希校验见 `reports/sft-tokenizer-download.json`，可复用 `scripts/download_sft_tokenizer.py` 下载必要文件。不能用字符数、别的模型或模拟 tokenizer 代替真实长度统计。

## 第一阶段：转换、去重、分组、划分和长度盘点

```powershell
& $DataPython -X utf8 -m lightagent.sft_data prepare `
  --manifest configs/sft-data-v1.json `
  --tokenizer $QwenTokenizer `
  --output data/sft_v1_inventory
```

等价 PowerShell 包装入口如下，默认 `inventory`，不会自动混合或训练：

```powershell
.\scripts\prepare_sft_v1.ps1 -PythonExe $DataPython `
  -Stage inventory -TokenizerDirectory $QwenTokenizer
```

该阶段读取三个 Hermes 文件、ToolACE 和已经生成的自建任务库。公开输入的固定 revision、文件 SHA256 必须与下载清单相符。自建任务库的验证报告必须通过、代码指纹必须匹配，任务、轨迹及已有公开 SFT 文件的 SHA256 必须与原验证报告相符。

公开记录解析或结构校验失败时，记录来源 ID、原始行、剔除阶段及原因；自建记录失败会停止整次准备，避免悄悄改变已有训练、验证、测试划分。输出目录已经存在时拒绝覆盖，应另选目录。写出失败时，`.preparing-*` 临时目录及 `FAILED.json` 用于排查，不作为成功数据集使用。

这一阶段**不按 9,000 条硬抽样**。输出中 `train.jsonl` / `validation.jsonl` 是完整盘点视图；`mode=inventory_not_target_sampled`，训练入口会要求先检查报告并显式执行第二阶段。

先查看报告：

```powershell
Get-Content -LiteralPath data/sft_v1_inventory/preparation-report.json -Raw -Encoding UTF8
```

重点查看：

- `raw_counts`、`stages`、`stage_counts_by_source`：原始、失败、转换成功、完全重复移除、唯一记录数量。
- `grouping`：前缀关系、Hermes 单轮/多轮家族、工具签名与调用结构、词面近重复关系、最大组、跨 split 组和完全重复数量。已知的 1,883 条关联由实际输入重新识别，不硬编码成“验证通过”的结果。
- `pool_by_split`：各 split 来源、主行为、重叠标签、token 的 min/mean/p50/p90/p95/p99/max，以及 >512、>4096 的数量。
- `full_views`、`overlength_records`、`overlength_ratio`、`content_quarantined`：正式视图实际保留与过滤情况。
- `self_expected_input_counts`、`self_retained`：输入应为训练 2,000、验证 300、三类测试各 200；若完全去重或长度限制使实际保留不足，必须报告并处理，不能复制或借用验证/测试记录补足。
- `local_smoke`：完整编码 <=512 token 的实际保留数量及来源/行为分布。允许为零，不删 schema、工具返回或用户后续消息来凑长度。

`full_max_tokens` 初始为 **32768**，是可调整的完整协议筛选上限，不是已确定的正式训练窗口。超长记录仍在 `pool/` 中追溯，在根目录正式视图中整条剔除；需要更大窗口或重新构造时，另建版本重新盘点。P1000 始终只使用 <=512 的子视图。

## 转换、分组和行为标签的规则

Hermes 优先使用记录已有 `tools`，将 `human/gpt` 映射为 `user/assistant`，解析 `<tool_call>` / `<tool_response>`。ToolACE 从 system 的工具定义区提取工具，解析 `[工具名(参数=值)]`。解析顺序是合法 JSON，再使用有长度、节点数和嵌套深度限制的 AST 字面量白名单；不会执行表达式，不使用 `eval()`。

规范化工具名时同步修改定义、调用和响应，并在元数据保存名称映射与原始调用 ID 映射。Schema 的 `dict/list/int/float/bool` 别名规范化为 JSON Schema 类型；调用参数保持原值，不把不符合类型的字符串猜测转换成数值。工具名碰撞、未知工具、非法 Schema、错误参数、重复调用 ID、孤立返回、未完成返回就进入下一轮、没有训练目标等均拒绝。

返回优先按显式调用 ID 匹配，其次按唯一工具名匹配。仅有一个待返回调用时可以接收未命名返回；多个调用绝不按列表位置猜配。同批同名工具没有明确 ID 时不能消歧，整条拒绝。错误返回和后续恢复保留。末尾停在 assistant 调用的单轮示范有效，标为 `call_generation_only`，不算反馈示范。

公开数据的原 system 主要包含来源专用工具输出协议，转换为项目已有的统一 `PUBLIC_SYSTEM`。工具定义完整放入 `tools`，原 system 的摘要、原始行 ID、输入文件哈希和固定 revision 保存在元数据，可回查原文。自建 system 和全部公开消息逐字复用重放导出，不缩减其工具或对话。长度统计包含实际输入的统一 system、完整工具定义、全部对话及返回。

完全去重使用规范化工具顺序、JSON 内容和调用 ID 的对话指纹。重复来源保存在保留项的 `provenance`，不丢失许可证、原始 ID 和映射。再以完整对话前缀连接单轮与多轮，以使用中的工具 Schema、调用结构和词面近重复请求连接公开数据家族，整个家族固定到同一 split 后才抽样。公开数据同一工具签名/调用结构会保守放在一起，可能产生较大的组，报告会披露；不会为了凑验证数量拆组。词面近重复使用确定性的 MinHash 候选和 Jaccard/文本相似度检查，不能视为语义去重的证明。

自建数据使用已有私有语义分组和原 split，不因共享八个工具而合成一个大组。任何完全重复或前缀关联导致已有自建 split 冲突都会停止，绝不自动把验证或测试迁入训练。每条自建轨迹都重放，要求终局成功、reward=1、公开导出匹配，并执行现有私有字段及 canary 泄漏检查。重放只验证本地模拟环境及结构化事实，不宣称验证了自由文本的一切语义。

行为可以多标签，统计配额只使用一个 `primary_behavior`：

| 主标签 | 依据 |
| --- | --- |
| `selection` | 合法工具调用，含末尾调用生成示范 |
| `feedback` | 工具返回后继续调用或回复的结构证据；公开数据未重新执行真实 API |
| `clarification` | 自建重放验证的缺参询问，或绑定指纹的人工审核证据 |
| `no_call` | 人工确认无需工具、工具不适用或合理停止 |
| `unreviewed_no_call` | 尚无充分证据；保留盘点，不参加目标抽样 |

公开反馈示范还会检查结构化最终结果与工具结果的直接冲突、未恢复错误后的明确成功声明、少数明确天气矛盾；可疑项隔离，审核通过前不进入正式视图。其余自由文本只完成启发式检查，不能标为真实 API 重执行通过或“语义全部验证”。

初始公开数据的澄清/不调用类别可能不足。禁止把所有问句或普通聊天自动塞入这两类。若需审核，复制盘点输出中的 `metadata/review-schema.example.json`，按对应元数据中的 source、source_id、fingerprint 填成 JSONL，然后将路径加入 `configs/sft-data-v1.json` 的 `review_files`，重新准备到新目录。审核必须记录审核者和具体证据。澄清还需要工具名、实际缺失的必需参数、问题消息索引，且对应工具不能已经在问题前猜参调用。不调用审核需要 `evidence.reason` 为 `tool_not_applicable`、`no_tool_needed` 或 `reasonable_stop`。

## 第二阶段：查看盘点结果后，手动选择混合比例

`configs/sft-mixture-v1.example.json` 只是起始政策：公开训练 7,000、独立公开验证 400，主行为 30%/50%/10%/10%。来源采用无放回加权排序，优先 Hermes 多轮、Glaive，再用 ToolACE 补充；Hermes 单轮上限为各公开 split 目标的 8%。来源权重是辅助偏好，不保证固定来源数量。

先根据真实供给调整政策，再执行：

```powershell
& $DataPython -X utf8 -m lightagent.sft_data mix `
  --dataset data/sft_v1_inventory `
  --policy configs/sft-mixture-v1.example.json `
  --output data/sft_v1
```

`mix` 只从第一阶段已固定 split 的合格公开记录中选择。每类先满足自身目标，供给不足则报告初始缺口，再把剩余额度分配给实际可用类别；总供给不足就输出真实较小规模。不会重标行为或重复复制。自建数据不抽样、不扩增，始终保留原有 split；同样遵守完整视图的长度与完全去重结果。公开训练 7,000 不包含自建训练 2,000；公开验证目标也不包含自建验证 300。

查看 `preparation-report.json` 的 `mixture_result`：请求数量、实际数量、总缺口、每类初始缺口、调整后比例、来源权重与单轮上限都有记录。若供给确实不足，记录调整后的比例和原因，再另建版本；不能从验证或测试补训练。

第二阶段输出如下。第一阶段盘点目录也保留相同的追溯结构：

```text
data/sft_v1/
  train.jsonl
  validation.jsonl
  test_id.jsonl
  test_composition.jsonl
  test_robustness.jsonl
  pool/                         # 此版本选择的完整、未截断轨迹；初始全集在 inventory 中
  metadata/
    train.jsonl                 # 与数据文件逐行对应，不传入模型
    validation.jsonl
    test_*.jsonl
    pool-*.jsonl
    local_smoke-train.jsonl
    local_smoke-validation.jsonl
    rejected.jsonl
    duplicates.jsonl
    tokenizer.json
  local_smoke/
    train.jsonl
    validation.jsonl
  preparation-report.json
```

模型 JSONL 每行严格只有 `tools` 和 `messages`。`function.arguments` 是对象，工具返回 `content` 是字符串。元数据通过行号与规范化指纹绑定，报告还记录数据/元数据文件 SHA256。不要单独重排、修改或拼接某一侧。

## 第三阶段：真实 Qwen tokenizer 与 mask 审计，仍不训练

```powershell
& $DataPython -X utf8 -m lightagent.sft_data audit-mask `
  --dataset data/sft_v1 `
  --tokenizer $QwenTokenizer `
  --random-count 32 --per-behavior 4 --seed 42
```

该命令检查训练、验证中的每种现有行为与附加标签，再加确定性随机抽查和**所有**本机短样本；缺失类别明确报告，不用第一条代表全部。审计复用训练模板，要求与长度阶段的 tokenizer 文件哈希、模板哈希和软件版本一致，确认：

- system、user、tool 返回没有监督 token；
- assistant 的调用名和参数字段、澄清问题、最终回复具有监督；
- 每个 assistant 片段中的实际 Qwen `<|im_end|>` 都参与监督；
- 再编码的完整长度与盘点一致，短样本不超过 512。

失败直接停止，不写“通过”。成功后写 `metadata/mask-audit.json`，在准备报告记录审计摘要，并绑定报告哈希。模板获取、前缀保持和结束 token 检查沿用项目原训练入口；上游说明见 [TRL Chat Templates](https://huggingface.co/docs/trl/chat_templates) 与 [TRL 模板实现](https://github.com/huggingface/trl/blob/main/trl/chat_template_utils.py)。

新候选配置 `configs/sft-v1-p1000.json` 接入这份独立元数据和短样本视图。训练加载会校验数据/元数据哈希、原始 split、来源、自建重放证明和当前 mask 审计；盘点视图或过期审计会拒绝。旧配置不会自动切换到新数据。本阶段不提供或调用 `--execute`，也不启动 Trainer。

512-token 短样本可能几乎只剩简单调用；即使之后冒烟成功，也只说明训练链路可用，不能证明完整多轮能力已训练。若短样本为空，应调整冒烟设计或硬件，不裁剪协议。

## 可选：由使用者执行已编写的单元测试

```powershell
& $DataPython -X utf8 -m pytest -q tests/test_sft_data.py tests/test_data_pipeline.py tests/test_training_contracts.py
```

这些用例覆盖受限解析、同名并行调用消歧、错误恢复、Schema 类型拒绝、单轮/多轮家族、自建 split 冲突、保守行为标签、抽样缺口和元数据错位等。本次执行共 39 项通过，包含真实本地 Qwen tokenizer 的并行工具返回检查；另对最终数据中的 471 条记录执行了真实 mask 审计。
