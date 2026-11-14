# SECO 逐样本分组统计

通用 Benchmark、NIAH 和 RULER QA 的 SECO 评测默认记录分组统计。每个样本在每个压缩比下有一条记录，写入对应的 `ratio_*/predictions.jsonl`。多卡运行时，统计随预测一起分片并合并。

## 字段与计数口径

| 字段 | 含义 |
| --- | --- |
| `id` | 样本标识，可与对应压缩比的预测文件关联 |
| `dataset` 或 `task` | 通用 Benchmark 使用 `dataset`，NIAH 和 RULER QA 使用 `task` |
| `ratio` | 当前样本使用的评测压缩比 |
| `group_count` | SECO 实际生成的上下文分组数 |
| `group_token_counts` | 按压缩输出顺序排列的各组 token 数，每组包含一个中心 token |

每条记录的 `group_count` 等于 `group_token_counts` 的长度。列表求和等于该样本实际参与分组的上下文 token 数。直接保留的 prompt、padding 和生成 token 均在计数范围之外。所有上下文 token 都是中心时，每组大小为 1。空上下文记录为 `0` 和 `[]`。

## Shell 变量与 Python 参数

| Shell 变量 | Python 参数 | 默认值 | 修改提示 |
| --- | --- | --- | --- |
| `RECORD_GROUP_STATS` | `--record-group-stats` / `--no-record-group-stats` | SECO 为 `1` | 将变量设为 `0` 可关闭记录。也可直接修改对应 SECO `50_benchmark.sh` 中的默认值 |
| `BENCHMARK_DIR` | `--benchmark-dir` | 必填 | 指向已完成的评测目录，该目录包含 `benchmark.json` 和各压缩比的合并预测 |
| `GROUP_STATS_OUTPUT` | `--output` | `$BENCHMARK_DIR/group_stats.jsonl` | 需要另存时指定新的 JSONL 路径。相对路径以仓库根目录为基准 |
| `PYTHON_BIN` | — | `python` | 指定 Linux 环境中的 Python 解释器 |

## 在 Linux 上运行

按对应实验的评测指南配置检查点与数据目录。以下示例显式开启统计，并指定输出目录供导出使用。

```bash
RECORD_GROUP_STATS=1 OUTPUT_DIR=artifacts/my_seco_run \
  bash scripts/benchmark/sh/seco/50_benchmark.sh

BENCHMARK_DIR=artifacts/my_seco_run \
  bash scripts/benchmark/sh/55_export_group_stats.sh
```

NIAH 或 RULER QA 使用各自的 `scripts/niah/sh/seco/50_benchmark.sh` 或 `scripts/ruler_qa/sh/seco/50_benchmark.sh`，运行完成后使用同一个导出脚本。NIAH 的 `51_benchmark_train.sh` 也沿用这个开关。

导出仅需 Python 标准库，结果为 UTF-8 JSONL，每行包含一个样本在一个压缩比下的身份字段和分组统计。旧结果或关闭记录的结果需开启统计后重新评测，再执行导出。重复导出时可通过 `GROUP_STATS_OUTPUT` 指定新文件。
