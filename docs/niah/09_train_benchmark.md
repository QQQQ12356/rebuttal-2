# 09 NIAH 训练集评测

训练集评测用于观察 checkpoint 对已训练样本的生成表现，例如比较不同压缩比下的训练集 EM。它直接读取预处理产物中的 `train.jsonl`，包括配置中实际保留的 synthetic、Cogito Bits 和 Cogito Bind 样本。生成时输入原有的 context 与 prompt token，参考答案只用于计分。独立测试集结果仍由 [05 测试网格](05_prepare_grid.md) 和 [06 留出测试](06_benchmark_plot.md) 产生。

## 1. Linux 执行顺序

1. 按 [01 环境](01_environment.md) 和 [02 数据下载](02_download.md) 准备 Linux 环境、模型与 raw 数据。
2. 按 [03 预处理](03_prepare_train.md) 生成 `train.jsonl` 与 `manifest.json`，记下实际 `PREPARED_DIR`。
3. 按 [04 训练](04_training.md) 使用该目录训练，选择相应的 `checkpoint-N`、`final` 或由该 checkpoint 合并导出的目录。
4. 设置 `EVAL_TRAIN=1` 并先用 `PLAN_ONLY=1` 核对训练数据版本、样本量和压缩比。

```bash
export PREPARED_DIR=artifacts/seco/Llama-3.2-1B-Instruct/niah/prepared/train_v1
export CHECKPOINT=artifacts/seco/Llama-3.2-1B-Instruct/niah/runs/实际训练目录/final
EVAL_TRAIN=1 PLAN_ONLY=1 MAX_TEST_SAMPLES=32 RATIOS='4 8' \
    bash scripts/niah/sh/seco/51_benchmark_train.sh

EVAL_TRAIN=1 MAX_TEST_SAMPLES=32 RATIOS='4 8' \
    bash scripts/niah/sh/seco/51_benchmark_train.sh
```

同样可以在 `50_benchmark.sh` 上用 `EVAL_TRAIN=1` 切换。专用 `51_benchmark_train.sh` 要求显式开启。执行独立网格评测时使用 `EVAL_TRAIN=0 bash scripts/niah/sh/seco/50_benchmark.sh`。训练集评测无需运行 `21_prepare_grid.sh`。

预检会校验训练数据和 checkpoint 元数据，读取训练样本，检查选中样本的预处理解码预算，并打印原始样本数、每个压缩比的样本数、总生成条数、选中样本的最大 encoder 长度和进程数。真实模型位置上限在加载模型后检查。预检无需加载模型或分配 CUDA。

## 2. 脚本变量与 Python 参数

| Shell 变量 | Python 参数 | 默认与用法 |
| --- | --- | --- |
| `EVAL_TRAIN` | `--eval-train` | `0`。设置为 `1` 显式选择训练数据，专用入口必须开启 |
| `CHECKPOINT` | `--checkpoint` | 填写实际 adapter 或 merged 目录。需要相同训练 manifest 的 NIAH checkpoint |
| `PREPARED_DIR` | `--prepared-dir` | `$NIAH_ROOT/prepared/train_v1`。改过训练数据路径时填写当时使用的目录 |
| `OUTPUT_DIR` | `--output` | Shell 自动生成 `$NIAH_ROOT/benchmarks/niah_train_*`。显式指定时使用新目录 |
| `RATIOS` | `--ratios` | `4 8 16 32`。用空格分隔，各比率须不同且不低于训练数据的 `min_ratio` |
| `MAX_TEST_SAMPLES` | `--max-test-samples` | 空则全量。正整数限制每个压缩比的样本总数，大于数据量时使用全量 |
| `BATCH_SIZE` | `--batch-size` | `1`。每张 GPU 的 batch size |
| `WORKERS` | `--workers` | `2`。每个进程的数据加载 worker 数，`0` 在主进程加载 |
| `WARMUP_BATCHES` | `--warmup-batches` | `1`。每张卡、每个压缩比在首批计时前的预热次数，`0` 关闭 |
| `MAX_NEW_TOKENS` | `--max-new-tokens` | `64`。贪心生成上限。长答案可增大，并检查下述预算 |
| `REPETITION_PENALTY` | `--repetition-penalty` | `1.0`。有限正数，`1.0` 表示不施加重复惩罚 |
| `PLAN_ONLY` | `--plan-only` | `0`。设置为 `1` 只检查并打印计划 |
| `CUDA_VISIBLE_DEVICES` | 由 CUDA 读取 | `0`。指定参与推理的物理 GPU |
| `NPROC_PER_NODE` | torchrun `--nproc_per_node` | `1`。单节点进程数，每张参与的 GPU 加载完整模型对 |
| 命令末尾传入 | `--model` | 空。adapter 基座迁移时指定相同权重的新路径 |
| 命令末尾传入 | `--attention` | 使用 checkpoint 设置。可选 `eager`、`sdpa`、`flash_attention_2` |
| 命令末尾传入 | `--no-merge` | 默认推理前合并。需要评测未合并 adapter 时传入 |

训练样本按准备时记录的种子进行可复现的蓄水池抽样。相同的数据版本和样本上限会选出相同样本，各压缩比使用同一批 ID。上限适用于整个训练集，包含多个来源时各来源的入选数可能不同。预检和评测都扫描完整 `train.jsonl`，设上限时仅保留抽中的 token 行及全量 ID 用于重复检查。

改压缩比或生成上限时，用最小评测压缩比核对解码预算。样本的压缩中心数、prompt token 数和 `MAX_NEW_TOKENS` 之和须在预处理 manifest 的 `max_decoder_tokens` 内，真实生成还受模型位置上限约束。多卡按 ID 列表的索引分片，每条样本只由一张卡生成；允许最后的小 batch 和空分片。

直接运行 Python 预检的等价示例：

```bash
python scripts/niah/py/seco/51_benchmark_train.py --eval-train --plan-only \
    --checkpoint "$CHECKPOINT" --prepared-dir "$PREPARED_DIR" \
    --output artifacts/seco/Llama-3.2-1B-Instruct/niah/benchmarks/train_diagnostic \
    --ratios 4 8 --max-test-samples 32
```

## 3. 输出与核对

| 产物 | 用途 |
| --- | --- |
| `benchmark.json` | `split=train`、实际参数、checkpoint 与训练 manifest 哈希、选中 ID 哈希和样本量 |
| `ratio_<R>/predictions.rank<N>.jsonl` | 每个 rank 的分片预测，逐批写入 |
| `ratio_<R>/runtime.rank<N>.json` | 各 rank 的 CUDA allocated 峰值 |
| `ratio_<R>/predictions.jsonl` | 按 ID 合并的逐样本预测、参考答案、来源、任务、输入长度与分数 |
| `ratio_<R>/metrics.json` | `seco-niah-train-metrics-v1` 指标，含整体、各来源与各任务的 EM、F1、hit rate 和样本数，以及计时和显存 |

EM、F1 和 hit rate 与留出测试使用相同计分函数，数值范围为 0–1。`accuracy` 等于 EM。多卡合并时会检查样本数和 ID 是否与选中样本完全一致。每个比例的整体分数按样本平均。训练样本没有统一的长度 × 深度网格，查看各比例的 `metrics.json`。现有 `60_score.sh` 和 `70_plot_heatmap.sh` 处理的是留出网格产物。

| 报错或现象 | 检查位置 |
| --- | --- |
| 要求 `EVAL_TRAIN=1` 或 `--eval-train` | 训练集入口需显式开启 |
| checkpoint 与训练 manifest 不一致 | 使用训练时的原始 prepared 目录或原样迁移的副本，检查 checkpoint 来源 |
| 训练数据哈希或 tokenizer 不一致 | 恢复与 manifest 匹配的 `train.jsonl` 或使用同一模型导出的 checkpoint |
| 解码预算超限 | 减少 `MAX_NEW_TOKENS` 或提高最小评测压缩比 |
| 生成到长度上限且答案不完整 | 核对答案长度，在预算内增加 `MAX_NEW_TOKENS` |
| 输出目录已存在 | 选择新 `OUTPUT_DIR`，或使用 Shell 自动生成的目录 |
