# 08 RULER QA 训练集评测

训练集评测直接读取训练时使用的 `train.jsonl` 和 `manifest.json`。模型使用原有上下文和问题生成答案，参考答案只用于计算 RULER substring score、EM 和 F1。指标反映模型在已见训练数据上的表现，应与独立测试集结果分开记录。

## 在 Linux 上运行

从仓库根目录执行。将 `CHECKPOINT` 改成实际训练产物目录，`PREPARED_DIR` 改成训练时传给 `30_train.sh` 的数据目录。适配器 checkpoint 和对应的合并导出均可使用。

```bash
export CHECKPOINT="$RUN_DIR/final"
export PREPARED_DIR=artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/prepared/train_v1

EVAL_TRAIN=1 MAX_TEST_SAMPLES=32 RATIOS='4 8' PLAN_ONLY=1 \
  bash scripts/ruler_qa/sh/seco/50_benchmark.sh

EVAL_TRAIN=1 MAX_TEST_SAMPLES=32 RATIOS='4 8' \
  bash scripts/ruler_qa/sh/seco/50_benchmark.sh
```

预检确认 `split` 为 `train`，并打印所选格子计数、全量行数 `prepared_test_samples`、实际行数 `test_samples` 和总生成次数 `generation_examples`。例如选中 32 行、两个压缩比时，总计生成 64 次。预检无需加载模型或启动 GPU 推理。

也可以直接调用 Python 入口。去掉 `--plan-only` 即在单卡上运行生成。

```bash
python scripts/ruler_qa/py/seco/50_benchmark.py \
  --eval-train --prepared-dir "$PREPARED_DIR" --checkpoint "$CHECKPOINT" \
  --output artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/benchmarks/train_qa_32 \
  --ratios 4 8 --max-test-samples 32 --plan-only
```

## 参数与修改位置

| Shell 变量 | Python 参数 | 默认值与作用 |
| --- | --- | --- |
| `EVAL_TRAIN` | `--eval-train` | `0` 评测独立测试网格。设为 `1` 评测 checkpoint 的训练数据。 |
| `PREPARED_DIR` | `--prepared-dir` | `$RULER_QA_ROOT/prepared/train_v1`。选择训练产物目录，仅训练模式读取。Python 默认以 `artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa` 为根。 |
| `TEST_DIR` | `--test-dir` | `$RULER_QA_ROOT/prepared/test_v1`。独立测试模式的数据目录。 |
| `MAX_TEST_SAMPLES` | `--max-test-samples` | 留空使用全量。设为正整数可限制每个压缩比的总行数，超过可用行数时取全量。 |
| `CHECKPOINT` | `--checkpoint` | 必填。目录中应包含训练实验元数据和模型产物。 |
| `OUTPUT_DIR` | `--output` | Shell 默认在 `$RULER_QA_ROOT/benchmarks` 下生成带时间戳和随机后缀的目录。Python 必填，指定尚未创建的目录。 |
| `RATIOS` | `--ratios` | 默认 `4 8 16 32`。Shell 中用引号包裹空格分隔的多个值。各值须有限、互异，且至少达到准备数据时的 `min_ratio`。 |
| `MAX_NEW_TOKENS` | `--max-new-tokens` | 默认 `32`。须为正整数且不超过清单的 `generation_budget`。训练行还须满足生成时的解码长度预算。 |
| `PLAN_ONLY` | `--plan-only` | 默认 `0`。设为 `1` 执行数据、checkpoint 和参数预检。 |

样本上限对两个任务和所有长度格子合计生效。脚本按任务 × 长度格子轮流选取固定子集，再分配到各 GPU，因此压缩比与 GPU 数量变化时复用同一子集。上限小于格子数时，部分格子会显示为 `NA`。默认训练目标长度为 4K、8K，实际长度格子取自训练清单。

优先在命令前设置变量。若需要长期调整默认值，可修改 `scripts/ruler_qa/sh/seco/50_benchmark.sh` 顶部对应变量，同时核对 Python 入口中直接调用时的默认值。训练集的配额和长度由 `configs/ruler_qa/prepare.json` 的 `train_samples_per_task` 和 `train_lengths` 控制。调整这些准备参数后生成新训练数据时，要用对应数据训练得到的 checkpoint 进行训练集评测。

## 输出与排查

- `benchmark.json` 记录 `split: train`、参数、checkpoint 和本次评测清单。沿用 `test_manifest` 字段存储清单，方便现有评分入口读取。
- 每个 `ratio_*` 目录包含 `predictions.jsonl` 和 `metrics.json`。可用 `60_score.sh` 重算分数，用 `70_plot_heatmap.sh` 生成图表，命令见 [05 评测](05_evaluation.md)。训练集图标题带 `[TRAIN]`，限量子集带 `[SUBSET]`。
- 如果训练清单与 checkpoint 不匹配，将 `PREPARED_DIR` 指向训练时实际使用的目录。脚本会核对清单和数据文件的 SHA-256，以及 tokenizer 指纹。旧的 RULER QA 训练产物只要包含这些现有字段即可直接使用。
- 如果报 `long_decoder`，降低 `MAX_NEW_TOKENS` 或调高最小评测压缩比，然后重新预检。原训练样本按参考答案长度预留解码空间，生成时使用独立的 token 上限。
- 如果遇到输出目录已存在，改用新 `OUTPUT_DIR`。跑独立测试时使用 `EVAL_TRAIN=0`，并用 `TEST_DIR` 选择测试网格。

GPU、batch、模型位置和注意力后端参数见 [06 参数](06_parameters.md)。
