# 2026-11-16 统一 SFT 数据执行结果

已完成数据处理、真实 tokenizer 长度统计、混合、mask 审计与训练读取入口预检查。未下载或加载模型权重，未创建 Trainer，未启动训练。

| 数据 | 训练 | 验证 | test_id | test_composition | test_robustness |
| --- | ---: | ---: | ---: | ---: | ---: |
| 公开示范 | 7,000 | 400 | 0 | 0 | 0 |
| 自建轨迹 | 2,000 | 300 | 200 | 200 | 200 |
| 最终完整视图 | 9,000 | 700 | 200 | 200 | 200 |
| <=512 token 短样本 | 383 | 23 | — | — | — |

原始输入共 23,195 条，其中公开 20,295、自建 2,900；解析、支持的格式或一致性检查剔除 3,787 条，转换后 19,408 条，完全重复移除 5 条，统一盘点池保留 19,403 条。所有自建轨迹重放成功、终局奖励为 1，原划分完整保留。分组后 12,822 个家族，最大组 276 条，跨 split 分组重叠为 0；清洗后识别 832 条有效 Hermes 单轮/多轮前缀关联。

公开数据尚无具备审核证据的澄清或无需调用记录。原 30%/50%/10%/10% 的候选比例调整为 **37.5% 调用生成、62.5% 反馈续执行**，没有重新标注普通聊天或重复复制数据。公开训练对应 2,625 条调用生成、4,375 条反馈续执行。自建训练中的 264 条澄清轨迹完整保留。实际政策为 `configs/sft-mixture-v1.json`；报告记录原目标中公开训练澄清和不调用各缺 700 条、公开验证各缺 40 条的情况。

| 来源 | 最终训练 | 最终验证 |
| --- | ---: | ---: |
| Hermes 多轮 | 718 | 46 |
| Hermes 单轮 | 77 | 4 |
| Glaive | 3,194 | 188 |
| ToolACE | 3,011 | 162 |
| 自建 | 2,000 | 300 |

完整盘点视图上限为 32,768 token，实际最长 4,357，无记录超出该上限。长度使用真实 Qwen3.5-2B tokenizer、TRL 匹配训练模板和完整 system/tools/messages 编码，显式读取 `input_ids`，不截断。所有自建轨迹均长于 512，短样本只含公开示范；本机冒烟不能代表自建多轮能力已经训练。

验证结果：

- 39 项测试通过，`pip check` 无依赖冲突。
- mask 审计 471 条，包括全部 406 条短样本、随机样本及训练/验证中的现有行为类别；assistant 调用参数、澄清、回复和结束 token 受监督，system/user/tool 被屏蔽。无需调用类别仍缺失，未伪造该类别的通过结论。
- 新 P1000 配置的数据读取入口已实际编码训练 128 条、验证 16 条，超长跳过 0 条；未加载模型或创建 Trainer。
- 公开 API 没有重新执行；公开反馈仍只有结构和启发式一致性证据。词面近重复分组也不是语义去重证明。

运行时修复了编码字典长度误计和真实 Qwen system/并行工具片段的审计边界问题。首份错误长度盘点已标记 `invalid_do_not_use`，未用于抽样。该旧数据目录 `data/sft_v1_inventory.invalid-lengths-20261116` 已纳入清理，报告与 tokenizer 信息归档至 `reports/archive/invalid-lengths-20261116/`，清理结果见 `reports/workspace-cleanup.json`。完整转换与重放证据经哈希校验后复用，对全部记录重新编码生成的有效数据池仍为 `data/sft_v1_inventory`。

主要产物：

- `data/sft_v1/train.jsonl`、`validation.jsonl` 与三类测试：仅含 `tools/messages`。
- `data/sft_v1/metadata/`：独立来源、许可、ID、分组、行为和 mask 审计。
- `data/sft_v1/local_smoke/`：383 条训练、23 条验证的完整短协议。
- `data/sft_v1/preparation-report.json`：各阶段计数、分布、长度、原目标缺口及分组重叠。
- `reports/sft-v1-execution.json`：完整执行摘要、实际软件版本、代码/配置哈希、范围和限制。
- `reports/sft-v1-loader-check.json`：不训练的读取入口检查。
- `reports/sft-tokenizer-download.json`、`reports/sft-data-environment.txt`：上游 tokenizer revision/哈希及独立 CPU 数据环境。
