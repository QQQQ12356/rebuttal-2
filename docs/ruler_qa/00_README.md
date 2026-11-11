# RULER Question Answering 实验

以下命令以 SeCo 为例。使用 COMI、ICAE 或 RAM 时，将脚本路径中的方法段替换为相应小写名称，并使用该方法对应的产物路径。统一步骤见 [多方法实验指南](../00_methods.md)。

本流程评估四种方法压缩后的长上下文问答表现，包含 `qa_1`（SQuAD 2.0）和 `qa_2`（HotpotQA distractor）。默认基座为 `meta-llama/Llama-3.2-1B-Instruct`。目录独立于 NIAH，共用训练、合并与评测代码。

| 顺序 | 指南 | 主要产物 |
| --- | --- | --- |
| 01 | [Linux 环境](01_environment.md) | Python、CUDA 和依赖 |
| 02 | [模型与数据下载](02_download.md) | 本地模型、四份原始 JSON |
| 03 | [数据预处理](03_preprocessing.md) | 训练 JSONL、测试网格和清单 |
| 04 | [训练与合并](04_training.md) | 双 LoRA、可续训检查点、可选合并模型 |
| 05 | [测试、评分与绘图](05_evaluation.md) | 预测、指标 JSON、PNG、PDF、CSV |
| 06 | [配置与脚本变量](06_parameters.md) | 默认值、调整位置和联动规则 |
| 07 | [小规模端到端验证](07_smoke_test.md) | 单 GPU 小样本验证命令 |
| 08 | [训练集评测](08_train_benchmark.md) | 训练集开关、样本上限、预检与排查 |

在 Linux 项目根目录执行。先用小规模验证跑通，再用默认配置跑完整网格。

```bash
bash scripts/ruler_qa/sh/seco/00_setup_environment.sh
bash scripts/ruler_qa/sh/seco/01_download_model.sh
bash scripts/ruler_qa/sh/seco/10_download_data.sh
bash scripts/ruler_qa/sh/seco/20_prepare_data.sh
SPLIT=test bash scripts/ruler_qa/sh/seco/20_prepare_data.sh

export RUN_DIR=artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/runs/qa_main_v1
export OUTPUT_DIR=artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/benchmarks/qa_main_v1
bash scripts/ruler_qa/sh/seco/30_train.sh
CHECKPOINT="$RUN_DIR/final" bash scripts/ruler_qa/sh/seco/50_benchmark.sh
BENCHMARK_DIR="$OUTPUT_DIR" bash scripts/ruler_qa/sh/seco/70_plot_heatmap.sh
```

训练使用原始训练 split，测试使用 NVIDIA RULER v1 QA 的公开 dev 来源。此流程用 Llama tokenizer、本项目的提示边界和采样规则构造长度网格，报告时可写作“SeCo RULER v1 QA（自定义采样与长度网格）”。[NVIDIA RULER v1 QA 任务定义](https://github.com/NVIDIA/RULER/blob/main/scripts/synthetic.yaml)

LLM-as-Judge 的脚本和 1–10 分口径见 [05 LLM-as-Judge](../05_llm_judge.md)。
