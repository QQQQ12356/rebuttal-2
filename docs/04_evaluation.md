# 04 生成、指标与报表

`50_benchmark.sh` 接受 `final`、`checkpoint-N` 或 `final_merged`。适配器推理默认先在内存合并，可用 `--no-merge` 保持 LoRA 模块。移动适配器到其他机器时，用 `--model` 指向同一基座版本。每个压缩比产生一份预测与指标。

```bash
method=comi
task=niah                         # benchmark / niah / ruler_qa
export CHECKPOINT=/path/to/run/final
export RATIOS='4 8 16 32'
export OUTPUT_DIR=/path/to/new/eval
bash "scripts/$task/sh/$method/50_benchmark.sh"

PREDICTIONS="$OUTPUT_DIR/ratio_4/predictions.jsonl" bash "scripts/$task/sh/$method/60_score.sh"
BENCHMARK_DIR="$OUTPUT_DIR" bash "scripts/$task/sh/$method/70_plot_heatmap.sh"  # NIAH 或 RULER QA
```

Benchmark 用 `70_export_excel.sh`。三个任务都可以先设置 `PLAN_ONLY=1` 查看样本量。各方法的 `50_benchmark.sh` 顶部可直接修改推理设置。训练集评测统一用 `EVAL_TRAIN=1`，并将 `PREPARED_DIR` 指向检查点对应的训练预处理目录。NIAH 的 `51_benchmark_train.sh` 会开启同一选项，使用该方法 `50_benchmark.sh` 中的设置。NIAH 与 RULER QA 还可用 [05 LLM-as-Judge](05_llm_judge.md) 对已有预测评分。

```bash
# 训练集诊断。四种方法和三个任务使用相同的参数名。
EVAL_TRAIN=1 MAX_TEST_SAMPLES=100 SHUFFLE=1 SEED=42 RATIOS='16 32' \
    CHECKPOINT=/path/to/run/final PREPARED_DIR=/path/to/prepared_train \
    bash scripts/niah/sh/comi/50_benchmark.sh
```

| 变量 | 作用 |
| --- | --- |
| `EVAL_TRAIN` | 默认 0 使用独立测试集，1 使用检查点的训练数据。训练集成绩用于诊断 |
| `MAX_TEST_SAMPLES` | 每个压缩比的全局样本上限。默认 0 使用全部。所有压缩比复用同一份选样 |
| `DATASETS` / `SAMPLES` | Benchmark 专用。前者指定数据集，后者限制每个数据集的样本数。全局上限在各数据集取样后应用 |
| `SHUFFLE` / `SEED` | 设 `SHUFFLE=1` 用种子控制随机选样。NIAH 与 RULER QA 网格子集按单元轮流取样 |
| `MODEL_PATH` / `ATTENTION` / `DTYPE` / `NO_MERGE` | 指定重定位后的基座、覆盖 attention 后端、选择推理精度、保留活跃 LoRA 分支。精度默认沿用 checkpoint，可选 bfloat16、float16、float32 |
| `TEMPERATURE` / `TOP_P` / `TOP_K` / `NUM_BEAMS` | sample 或 beam 的专用参数。适用关系见 [解码配置](01_decoding.md) |

| 产物／参数 | 含义 |
| --- | --- |
| `benchmark.json` | 数据、模型或检查点身份、解码参数及并行环境 |
| `ratio_<R>/predictions.jsonl` | 逐样本预测、参考答案、生成时延及压缩记忆长度上界 |
| `ratio_<R>/metrics.json` | 任务确定性指标，例如 EM、F1 或 RULER substring score |
| `RATIOS` | 待测名义压缩比例。RAM 的实际比例受原文保留段影响，ICAE 的前缀另有一个 LM 标记 |
| `MAX_NEW_TOKENS` | NIAH 默认 64，RULER QA 默认 32。Benchmark 用 `QA_MAX_NEW_TOKENS=64` 与 `NEWS_MAX_NEW_TOKENS=512` |
| `BATCH_SIZE`、`NPROC_PER_NODE` | 每卡批大小和进程数。每进程加载完整模型，增加卡数分担样本 |
| `DECODING` | 默认 greedy，可改为 sample 或 beam。对应参数见 [解码配置](01_decoding.md) |

比较方法时固定 prepared 清单、ratio、样本数量、解码设置及基座 commit，并记录 `keep_ratio` 等影响实际前缀长度的设置。固定随机种子并复用相同测试数据，可以保持样本与网格单元对齐。
