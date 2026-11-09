# SeCo 实验指南

Benchmark、NIAH 和 RULER QA 共用 SeCo 模型、语义压缩算法及训练与合并实现。默认基座是 `Qwen/Qwen3-4B-Instruct-2507`。在 Linux 上执行实验命令，Windows 用于编辑和静态检查。

| 流程 | 实验指南 | 脚本 | 配置 |
| --- | --- | --- | --- |
| Benchmark | [五数据集训练、四数据集评测与 Excel 导出](docs/benchmark/00_README.md) | `scripts/benchmark/{py,sh}/` | `configs/benchmark/` |
| NIAH | [大海捞针训练、网格评测与热力图](docs/niah/00_README.md) | `scripts/niah/{py,sh}/` | `configs/niah/` |
| RULER QA | [SQuAD 与 HotpotQA 长上下文训练、评测和热力图](docs/ruler_qa/00_README.md) | `scripts/ruler_qa/{py,sh}/` | `configs/ruler_qa/` |

共享实现位于 `seco/`，共享 ZeRO 配置位于 `configs/zero*.json`。实验用例分别位于 `tests/benchmark/` 和 `tests/niah/`，共享模型验证和原始压缩基线位于 `tests/`。原始 SeCo 脚本保存在 `legacy/` 供核对。

在项目根目录执行各指南中的命令。已有 Benchmark 命令中的 `scripts/py/` 和 `scripts/sh/` 分别改用 `scripts/benchmark/py/` 和 `scripts/benchmark/sh/`。已生成的数据、检查点、评测结果和日志仍使用原来的 `artifacts/` 与 `logs/` 路径。

## 评测解码参数

三个评测的 `50_benchmark.sh` 都接受下列 CLI 参数，并将它们转发给 Python 入口。NIAH 和 RULER QA 的训练集评测也使用同一套选项。评测参数记录在输出目录的 `benchmark.json`。

| 参数 | 含义 |
|---|---|
| `--decoding greedy` | 默认。每步选择得分最高的 token。 |
| `--decoding sample` | 启用采样，可设置 `--temperature`、`--top-p`、`--top-k`，默认分别为 1.0、1.0、0（禁用 top-k 截断）。 |
| `--decoding beam --num-beams 4` | 启用 beam search，默认 beam 宽度为 4，返回一条结果。 |
| `--min-new-tokens` | 默认 0。生成达到指定长度前抑制 EOS。 |
| `--max-new-tokens` | NIAH 默认 64，RULER QA 默认 32。Benchmark 用 `--qa-max-new-tokens` 和 `--news-max-new-tokens`，默认分别为 64 和 512。上限须符合预处理时的生成预算。 |
| `--repetition-penalty` | 重复惩罚，默认 1.0。 |
| `--seed` | 随机种子，Benchmark 和 RULER QA 默认 42，NIAH 默认使用数据种子。Benchmark 也用此种子选择评测样本。 |

在 Linux 项目根目录运行，例如对 RULER QA 启用采样：

```bash
CHECKPOINT=/path/to/checkpoint bash scripts/ruler_qa/sh/50_benchmark.sh \
  --decoding sample --temperature 0.7 --top-p 0.9 --max-new-tokens 32 --seed 42
```
