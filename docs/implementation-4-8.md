# 第 4–8 部分实施与验收

项目：**面向智能穿戴场景的 Qwen3.5-2B 多轮工具调用 Agent 后训练与评测**。框架保持 TRL。当前交付工程能力与运行入口，自建训练/验证/测试数据集、实际模型训练与公共基准实验暂未开展；既有 20 个工程回归场景不扩充成训练集。

## 4．可执行任务环境

`environment.py` 提供 14 个工具，其中时钟、时间解析和创建后回读服务于执行核验。新增日程修改、删除、活动目录、偏好读取和受控用户询问。所有工具本地执行，不连接外部账户。

- 修改：先检查冲突、revision 与提醒时间，再一起提交事件和提醒。失败时两者均保持原状。
- 删除：删除指定事件并清除关联提醒；重复同一幂等请求返回同一结果，删除 ID 不复用。
- 规划：天气和活动工具只返回事实；活动选择由策略决定。
- 隔离：每次 rollout 从同一个显式场景创建深拷贝，不沿用上一条轨迹的写入。
- 用户模拟：DeterministicUser 仅回答 requested_slots 指出的必要缺失字段；重复、无关或未询问的信息不自动补充。
- 私有目标：EpisodeSpec 保存场景、目标和可选用户回答。策略只看到用户文本、工具 schema 和实际返回。

难度 L1–L3 可通过查询和组合工具表达；L4 使用澄清/user_ask；L5 使用故障注入与恢复；L6 用 Session 在明确的新用户意图下共享状态并执行修改/撤销。规则策略仍是原条件运动任务参考，不是所有 L1–L6 的通用解题器。

GoalSpec 可以声明事件、提醒、天气查询、待删除对象、应新增对象数和保护约束。校验器接受不同有效路径，例如删除后可用 calendar_get 的 not_found 或覆盖原时段的 calendar_list 核验，不要求复现参考调用序列。

需要运行一个外部明确提供的任务时：

```powershell
.\.venv\Scripts\python.exe -m lightagent.episode --spec PATH_TO_EPISODE.json --backend openai --output outputs/episode.json
```

EpisodeSpec 的 schema 可在 Python 中调用 EpisodeSpec.model_json_schema() 查看。项目没有生成任务数据的命令，也没有创建自建训练文件。未来外部任务进入训练时必须已经有 family_id 与 train/validation/test 划分；同一 family 不可跨训练和验证。ID、组合泛化和鲁棒性标签通过私有 test_category 表达，本轮不填充这些测试集。

## 5．公开示范转换与 TRL SFT

只支持 xLAM 和 ToolACE 的用户已准备原始文件。未下载全量公开数据。xLAM 可能需要用户在平台接受访问条件；不绕过授权。

转换流程：JSON/Schema 校验 → 工具名规范化与碰撞检查 → AST 解析纯字面量 → 调用/返回配对 → 去重 → 按使用的工具签名分组划分 → 组间轮转抽样 → 可选真实 tokenizer 长度过滤 → 输出来源、许可、revision、转换版本和拒绝原因。

规范化后的训练格式是 TRL 使用的 tools **列表**和标准 messages：assistant 的 tool_calls、tool 角色的 tool_call_id，参数为对象。没有沿用其他框架的 tool_call/tool_response 角色。ToolACE 中不能可靠解析或配对的记录剔除，不执行 eval() 或数据中的代码。同批同名多调用若无法消歧会拒绝。xLAM 的末尾 assistant 调用可作为 SFT 目标，但不伪造工具观察。

分组只使用调用工具的 schema，忽略参数取值、问题措辞和无关工具；相同签名组全部进入同一 split。它是保守防泄漏策略，可能让小数据的验证集为空；此时训练入口报错，不能把测试集拿来补齐。当前不是跨语言改写或语义近重复的完整去重系统。

复制 configs/public-data.example.json，填入原始文件路径与真实的 **40 位数据 commit**，再运行：

```powershell
.\.venv\Scripts\python.exe -m lightagent.data_pipeline --manifest PATH_TO_PUBLIC_MANIFEST.json --output data/public
```

可用独立训练环境的 --tokenizer LOCAL_MODEL_PATH 检查完整轨迹长度；不指定时报告 length_checked=false，不会用字符数假装 token 数。实际 SFT 入口会再次对全部记录检查，遇到超长轨迹直接拒绝，禁止从中间截断工具协议。

候选 SFT 配置是语言侧 LoRA rank 16、BF16、学习率 1e-4、micro batch 1、累积 16、1 epoch，文本长度预算 4096，关闭思考和 packing。只选择语言侧 Linear 层，检查 trainable 参数全部为语言侧 LoRA，视觉模块保持冻结。超参尚未实测。

```powershell
# 本机可运行：仅打印计划与缺失输入
.\.venv\Scripts\python.exe -m lightagent.training --config configs/sft.json

# 以下仅在准备好模型、公开数据、Python >=3.11 训练环境后运行
python -m lightagent.training --config configs/sft.json --check-mask
python -m lightagent.training --config configs/sft.json --execute
```

mask 检查使用 TRL 的 Qwen 匹配训练模板：要求模板保持前缀并监督 assistant 结束 token；检查用户、系统和工具返回均屏蔽，而 assistant 调用和最终回复均有监督 token。输出 mask-audit.json，前 16 条包含可读的监督片段，仍应人工审查。当前测试使用简化 tokenizer 验证审计逻辑，**尚未通过真实 Qwen tokenizer 验收**。

训练读取公开训练与验证分组，不读取 test。以验证 loss 选基础检查点；实际任务能力还需独立评测。保存 final-adapter 和合并的 final-merged，后者作为 GRPO 起点。

## 6．环境交互式 GRPO

入口绑定 TRL 1.18.0 的 GRPOTrainer.environment_factory。每行外部任务以 episode_json 私下交给 reset；同组的同一任务从一致初态出发，环境互不共享。真实工具结果进入原生工具消息，使用 TRL 的工具 token mask，不把工具观察当作模型生成监督。

没有自建任务集，也没有自动将公开 SFT 返回值转换成可执行环境。episodes_file 当前有意指向尚未准备的外部文件；只有配置路径不能实际训练。

```powershell
# 当前仅执行配置计划
.\.venv\Scripts\python.exe -m lightagent.training --config configs/grpo.json
```

配置为 SFT 合并权重 + 新 RL LoRA，G=4，学习率 5e-6，beta=0.02，20 次更新，vLLM colocate。单轮最多 256 个新 token；整条完成轨迹最多 2048 token，包含工具结果；工具调用轮数最多 12。预算分开控制，并检查单轮 cap 是否实际生效。上下文先设置 8192，以容纳完整工具 schema；实际输入仍要经过“prompt + 轨迹预算”检查，不能直接声称 4k 能装下所有工具。

单轮 cap 使用锁定版本的 _generate_single_turn 适配，升级 TRL 时必须重新核查。显存利用率 0.35、sleep mode 和其余参数都是候选值，未在 A100/vLLM 上实测。可按预算将 G 降到 2，同时保持 generation batch 可整除。

可选的 validation_episodes_file 支持外部验证任务，并拒绝训练/验证 family 重叠。正式选择检查点或奖励权重需要验证集；当前配置是 20 步冒烟起点，不进行正式选择。

训练时保存：

- run-manifest.json：实际包版本、GPU/驱动、命令、完整配置、模板 hash、LoRA 目标层、峰值分配显存。
- trainer-metrics.json：TRL 日志中的 loss、KL、梯度、长度等实际返回指标，不伪造未返回字段。
- rollout-rewards.jsonl：每条轨迹的工具日志、终态、奖励细项。
- group-diagnostics.jsonl：奖励标准差、常量组、全失败/全成功组、非法及冗余调用。
- 训练前后参数 fingerprint：若 LoRA 参数未变化则报错，不能声称完成有效更新。

当前没有运行 Trainer，因此这些训练产物尚不存在。

## 7．可验证奖励与人工 Rubric

固定版本为 manual-v1：终态成功 S ∈ {0,1}，过程分 P ∈ [0,1]，成本 C ∈ [0,1]。

```text
result_only:      R = S
result_process:   R = S + 0.10 P - 0.05 C
```

权重被限制在小范围，失败轨迹不能靠格式或过程奖励超过完整成功轨迹。P 来自明确目标谓词，不由在线模型随意变动；C 包括非法调用和同结果的连续冗余查询。故障后的合理重试不会一律受罚。

检查事件/时间/时长、提醒关联与提前量、删除后缺失、受保护日程、创建/修改后回读，以及结构化反馈与真实结果的一致性。成功必须同时满足目标、证据和策略原始 success；Runtime 将虚报修正成 failed，并不会让其得到成功奖励。

自由文本的自然程度、解释质量和全部语义事实尚未由 Judge 判定。当前没有部署 Judge、拟合奖励或自动 Rubric。未来自动 Rubric 应先由任务结构生成固定条目、做可执行性检查与人工抽查；Judge 只离线辅助校准，不代替确定性状态校验。

显式任务和轨迹可以重放并复算：

```powershell
.\.venv\Scripts\python.exe -m lightagent.replay --spec PATH_TO_EPISODE.json --artifact outputs/episode.json
```

工具返回或终态与重放不一致时拒绝评分；旧轨迹缺少 raw_finals 时不能拿修正后反馈替代原始模型反馈。

## 8．公共基准、执行评测与效率

公共基准通过 EvalScope 自己的协议访问模型服务，不经过只支持单步一个调用的演示 EndpointPolicy，因此可以评测 BFCL parallel。七类固定范围是 simple、multiple、parallel、irrelevance、multi_turn_base、multi_turn_miss_param、multi_turn_miss_func。

```powershell
# 本机只打印计划，不拉取基准
.\.venv\Scripts\python.exe -m lightagent.benchmarks --config configs/bfcl-smoke.json

# 独立评测环境，准备好服务后再执行
python -m lightagent.benchmarks --config configs/bfcl-smoke.json --execute
python -m lightagent.benchmarks --config configs/bfcl-smoke.json --formal --execute
```

默认 limit=5 是接口冒烟。--formal 去掉 limit，运行这些固定子集，**仍不是 BFCL 总分**。实际类别样本数和得分必须读取 EvalScope 原报告，计划中保留 null，不假定执行成功。公共题不进入 SFT 管线。

场景执行层使用既有 20 个手写回归场景与 GoalSpec 校验，新增恢复率、澄清完成率、冗余调用、超限、模型 HTTP 时间、工具时间和 token usage 字段。不具备参考动作标签的“工具选择正确率”保留 null；不把 Schema 合法率当作参数语义正确率。失败场景必须真的出现相应错误证据；随机说失败不会获得正确处理分。

```powershell
.\.venv\Scripts\python.exe -m lightagent eval --output outputs/eval-rule.json
.\.venv\Scripts\python.exe -m lightagent eval --backend openai --label base --output outputs/eval-base.json
```

outcome_accuracy 包括正确澄清与正确失败，task_completion_rate 只计真实完成；两者不能混为模型成功率。澄清完成率仅在有补全答案的缺参场景统计，恢复率仅在既有可恢复故障场景统计。样本很少，不能代替泛化测试。

效率的完整任务时间来自 Runtime；纯请求测量可用流式采集：

```powershell
.\.venv\Scripts\python.exe -m lightagent.profiling --payload PATH_TO_CHAT_REQUEST.json --output outputs/profile.json
```

TTFT 从第一个内容/工具调用增量开始计，忽略空的角色增量；服务未返回 token usage 则为 null。HTTP 耗时含网络与排队，不等于纯 GPU 前向。模型峰值显存需在服务端采集，远程调用时本地不猜测该数值。模拟工具耗时不能称为穿戴设备实测响应速度。

实验组与种子见 configs/experiments.json。原始模型、公开 SFT、两种 GRPO 奖励与零 KL 对照均标记待运行；公开+自建 SFT 明确暂缓。没有任何模型能力提升或公共基准成绩被填入报告。

## 环境与验证

本地轻量环境：Python 3.10 + Pydantic + tzdata + jsonschema。独立训练环境建议 Linux、Python >=3.11，先安装匹配 CUDA 的 PyTorch，再安装项目和 requirements/train.txt；colocate 还需另行确定兼容 vLLM。评测建议独立安装 requirements/evaluation.txt，避免与训练依赖互相升级。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

本轮只运行这些轻量测试、既有规则回归和默认计划命令，没有运行公开数据清洗、下载模型、构建自建数据或真实训练。

依据：[TRL v1.18.0 源码](https://github.com/huggingface/trl/tree/v1.18.0)、[SFT 文档](https://huggingface.co/docs/trl/sft_trainer)、[GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)、[EvalScope BFCL-v3 文档](https://evalscope.readthedocs.io/en/latest/benchmarks/bfcl_v3.html)、[ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)、[xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)。
