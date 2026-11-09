# 01 评测解码配置

在 **Linux 项目根目录**运行下列命令。三个评测共用 [Python 解码选项](../seco/generation.py)，Bash 启动脚本顶部各有一组可编辑的变量。

| 评测 | Bash 启动脚本 | Python 入口与任务说明 |
|---|---|---|
| Benchmark | [50_benchmark.sh](../scripts/benchmark/sh/50_benchmark.sh) | [50_benchmark.py](../scripts/benchmark/py/50_benchmark.py) · [05 Benchmark 评测](benchmark/05_benchmark.md) |
| NIAH | [50_benchmark.sh](../scripts/niah/sh/50_benchmark.sh) | [50_benchmark.py](../scripts/niah/py/50_benchmark.py) · [06 NIAH 评测](niah/06_benchmark_plot.md) |
| RULER QA | [50_benchmark.sh](../scripts/ruler_qa/sh/50_benchmark.sh) | [50_benchmark.py](../scripts/ruler_qa/py/50_benchmark.py) · [05 RULER QA 评测](ruler_qa/05_evaluation.md) |

## 1. 修改方式

- **长期修改**。在对应 `50_benchmark.sh` 顶部修改变量的默认值。例如将 `DECODING="${DECODING:-greedy}"` 中的 `greedy` 改为 `sample`，再将 `TEMPERATURE="${TEMPERATURE:-}"` 改为 `TEMPERATURE="${TEMPERATURE:-0.7}"`。
- **单次覆盖**。在 Bash 命令前设置环境变量，或在脚本名后追加 Python CLI 参数。相同选项在命令末尾显式指定时，以末尾的值为准。
- **直接运行 Python**。沿用各任务文档中的 Python 启动命令，追加表中的 CLI 参数。NIAH 的训练集入口 `51_benchmark_train.py` 与测试入口共用参数实现，RULER QA 通过 `--eval-train` 切换到训练集。
- **切换模式**。采样变量只在 `sample` 模式下填写，`NUM_BEAMS` 只在 `beam` 模式下填写。脚本会转发所有非空的模式专用变量。切换模式时清空上一模式的专用变量，避免 Python 报参数冲突。如果脚本已为某个专用变量设置了非空默认值，需要在脚本中恢复为空。

## 2. 通用生成变量

| Bash 变量 | Python 参数 | 默认值与作用 |
|---|---|---|
| `DECODING` | `--decoding` | `greedy`。贪心每步选得分最高的 token。`sample` 从分布采样。`beam` 保留多条候选路径并返回一条结果。 |
| `TEMPERATURE` | `--temperature` | 脚本留空，采样时为 `1.0`。要求有限且大于 0。小于 1 会使采样集中于高分 token。 |
| `TOP_P` | `--top-p` | 脚本留空，采样时为 `1.0`。范围为 `(0, 1]`。较小值缩小累计概率覆盖的候选集合。 |
| `TOP_K` | `--top-k` | 脚本留空，采样时为 `0`。非负整数。正值限制候选 token 数，0 关闭 top-k 截断。 |
| `NUM_BEAMS` | `--num-beams` | 脚本留空，beam 模式下为 `4`。整数且至少为 2。提高宽度会增加生成开销。 |
| `MIN_NEW_TOKENS` | `--min-new-tokens` | `0`。达到指定新 token 数前抑制 EOS。须介于 0 与当前任务的生成上限之间。 |
| `REPETITION_PENALTY` | `--repetition-penalty` | `1.0`。要求有限且大于 0。大于 1 会惩罚重复 token，可能影响需要复现原词的答案。 |
| `SEED` | `--seed` | Benchmark 和 RULER QA 为 `42`。NIAH 留空时使用当前数据的种子。生成进程在基础种子上加 rank，Benchmark 的可选样本重排也使用基础种子。 |

## 3. 各任务的长度和运行变量

| 评测 | 生成上限变量 | 修改提示 |
|---|---|---|
| Benchmark | `QA_MAX_NEW_TOKENS=64`、`NEWS_MAX_NEW_TOKENS=512` 对应 `--qa-max-new-tokens`、`--news-max-new-tokens` | QA 数据集和 MultiNews 分开设置。所选数据集的上限不得超过预处理的 `test_generation_budget`。 |
| NIAH | `MAX_NEW_TOKENS=64` 对应 `--max-new-tokens` | 测试网格受 `grid.generation_budget` 约束，训练集评测还需符合该数据的 decoder 预算。 |
| RULER QA | `MAX_NEW_TOKENS=32` 对应 `--max-new-tokens` | 不得超过所选 prepared manifest 的 `generation_budget`，并须符合 decoder 长度预算。 |

- `CHECKPOINT` 指定 adapter checkpoint 或已合并模型。数据目录通过 Benchmark 的 `PREPARED_DIR`、NIAH 的 `GRID_DIR` 或 RULER QA 的 `TEST_DIR` 指定。NIAH 与 RULER QA 使用 `EVAL_TRAIN=1` 时，`PREPARED_DIR` 指向对应训练集。
- `CUDA_VISIBLE_DEVICES` 选择 GPU，`NPROC_PER_NODE` 控制进程数，`BATCH_SIZE` 是每进程 batch 大小。调高 beam 宽度时，可按显存调整 batch 大小。
- `OUTPUT_DIR` 指定新的输出目录。每次运行的 CLI 参数保存在 `benchmark.json` 的 `arguments` 中，实际传给生成的参数另存于 `generation`，Benchmark 按数据集存于 `generation_by_dataset`。
- `PLAN_ONLY=1` 可用于 NIAH 和 RULER QA 的输入与预算预检。预检不加载模型生成答案。

## 4. Linux 命令示例

```bash
# Benchmark，默认贪心，单次调整 QA 长度
CHECKPOINT=/path/to/benchmark-checkpoint QA_MAX_NEW_TOKENS=48 \
  bash scripts/benchmark/sh/50_benchmark.sh

# NIAH，使用环境变量启用 beam search
CHECKPOINT=/path/to/niah-checkpoint DECODING=beam NUM_BEAMS=4 \
  bash scripts/niah/sh/50_benchmark.sh

# RULER QA，使用 CLI 启用采样
CHECKPOINT=/path/to/ruler-checkpoint bash scripts/ruler_qa/sh/50_benchmark.sh \
  --decoding sample --temperature 0.7 --top-p 0.9 --seed 42
```

排查过早结束时，可以固定检查点、数据、压缩比和解码方式，对照 `MIN_NEW_TOKENS=0` 和 `MIN_NEW_TOKENS=2`。最小长度按 token 计数，一个词可能包含多个 token。对照时同时看预测文本和正确率，短答案任务中有些正确答案本身就很短。采样复跑还应固定 batch 大小、进程数和 warmup 设置，因为这些会改变随机数的消费顺序。
