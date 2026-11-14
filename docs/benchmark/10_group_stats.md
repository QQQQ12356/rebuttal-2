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

## 通用 Benchmark 的组大小与 F1 分析

分析脚本直接读取通用 Benchmark 评测目录下的合并预测文件。输入支持 NQ（`naturalqa`）、HotpotQA（`hotpotqa`）、2WikiMQA（`2wikimqa`）、NarrativeQA（`narrativeqa`）和 MultiNews（`multinews`）。当前默认测试数据注册表包含其中四项。要纳入 NarrativeQA，需使用包含 `narrativeqa` 的已预处理测试数据和评测结果。脚本会提示当前 run 缺少的数据集。

```bash
BENCHMARK_DIR=artifacts/my_seco_run \
  bash scripts/benchmark/sh/56_analyze_group_f1.sh
```

| Shell 变量 | Python 参数 | 默认值与用途 |
| --- | --- | --- |
| `BENCHMARK_DIR` | `--benchmark-dir` | 必填，指向包含 `benchmark.json` 和 `ratio_*/predictions.jsonl` 的通用 Benchmark 目录 |
| `GROUP_F1_OUTPUT_DIR` | `--output-dir` | 默认在 run 下新建 `group_f1_analysis/`，重复分析时指定新目录 |
| `BOOTSTRAP_RESAMPLES` | `--bootstrap-resamples` | 每个数据集、压缩比及指标使用 1000 次样本重抽，设为 `0` 可跳过区间估计 |
| `BOOTSTRAP_SEED` | `--seed` | 默认 `42`，控制 bootstrap 随机种子 |
| `PYTHON_BIN` | — | 默认 `python`，使用 Linux Python 3.10+ 标准库 |

- `samples.csv` 每行对应一个样本和压缩比，保存原始 F1、组数、参与分组的 token 总数、组大小 CV（总体标准差除以均值）和最大组大小相对均值。组数小于 2 的样本保留在表中，两个不均衡指标留空。
- `correlations.csv` 按数据集、压缩比和指标输出 Spearman 相关系数 `spearman_rho`，以及控制上下文 token 数秩后的偏相关 `length_adjusted_rho)。并列值使用平均秩。负相关表示较不均衡的样本倾向于获得较低 F1。F1 沿用评测的 0–1 尺度，MultiNews 的 F1 表示 token 重合。
- `adjusted_ci95_low/high` 是偏相关的分层样本 bootstrap 百分位区间，假设同一层内样本相互独立。常量、样本量不足、完全共线或有效重抽不足时，相应结果留空。区间用于描述关联的不确定性。因果结论需结合固定压缩预算的配对消融。
