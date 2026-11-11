# 长上下文压缩实验

SeCo、COMI、ICAE 和 RAM 共用 Benchmark、NIAH 与 RULER QA 的数据、训练和评测流程。四种方法默认使用 `meta-llama/Llama-3.2-1B-Instruct`。在 Linux 上执行实验命令。

| 步骤 | 指南 |
| --- | --- |
| 00 | [方法、代码与目录](docs/00_methods.md) |
| 01 | [Linux 环境和模型下载](docs/01_environment.md) |
| 02 | [数据下载与预处理](docs/02_data.md) |
| 03 | [训练、恢复与合并](docs/03_training.md) |
| 04 | [生成、指标与报表](docs/04_evaluation.md) |
| 05 | [NIAH / RULER QA 的 LLM-as-Judge](docs/05_llm_judge.md) |
| 06 | [Shell 变量、Python 参数和 JSON 配置](docs/06_parameters.md) |

| 任务 | 专项指南 |
| --- | --- |
| Benchmark | [五数据集训练、四数据集评测与 Excel 导出](docs/benchmark/00_README.md) |
| NIAH | [检索训练、长度×深度网格与热力图](docs/niah/00_README.md) |
| RULER QA | [SQuAD 与 HotpotQA 长上下文训练、评测和热力图](docs/ruler_qa/00_README.md) |
| ICAE | [预训练、任务微调和专用参数](docs/icae/00_README.md) |
| ICAE | [预训练、任务微调和专用参数](docs/icae/00_README.md) |
