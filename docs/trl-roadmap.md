# TRL 路线与验证边界

当前已实现第 4–8 部分的代码、配置和轻量验证，自建数据集不开展；详细操作见 [实施说明](implementation-4-8.md)。

| 模块 | 实现 | 尚需真实环境验证 |
| --- | --- | --- |
| SFT | SFTTrainer、语言侧 LoRA、公开数据校验、原生模板 assistant mask 审计 | Qwen tokenizer、真实样本 mask、GPU loss/更新 |
| GRPO | GRPOTrainer.environment_factory、独立环境、单轮/全轨迹预算、唯一奖励源 | 外部可执行任务、vLLM colocate、实际更新及资源消耗 |
| 奖励 | 固定人工 Rubric、终态谓词、有界过程与成本、原始反馈核验、复算 | 验证集权重选择、结果/过程消融 |
| 评测 | 工程回归、七类 BFCL 计划与启动、HTTP 效率采集 | 原模型与各训练模型的真实分数 |

本机运行环境是 Python 3.10，训练另用 Python >=3.11。2026-11-16 核查的候选版本为 TRL 1.18.0、Transformers 5.23.0、PEFT 0.21.3；它们尚未在本机 GPU 上联合验证。requirements/train.txt 是候选主依赖列表，不是已经验证的 CUDA/vLLM 或传递依赖锁。

实现依据锁定的 [TRL v1.18.0 源码](https://github.com/huggingface/trl/tree/v1.18.0)，避免混用 main 文档中的新接口。该版本允许数据行经 reset(**kwargs) 配置各 rollout；返回的字符串会追加到最后一个用户消息。本适配器在收到 episode_json 时返回 None，只使用已构造的公开请求，不重复请求或泄露私有字段。

get_reward() 会被 TRL 作为额外奖励源加入。本项目适配器故意不实现该方法，只通过一个 reward function 评分，防止重复加奖励。未来若迁移到其他版本，先检查方法发现、reset、响应解析和 token mask，再调整版本锁。

GRPO 基于合并后的 SFT 权重再加新的 RL LoRA。这样禁用 RL adapter 时参考的是 SFT 模型，而不是未经 SFT 的原模型。beta=0 与小非零值、仅结果与结果加过程的对照在 configs/experiments.json 中列为待运行实验。

参考：[SFT 官方文档](https://huggingface.co/docs/trl/sft_trainer)、[GRPO 官方文档](https://huggingface.co/docs/trl/grpo_trainer)、[TRL 版本元数据](https://pypi.org/pypi/trl/json)。
