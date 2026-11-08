# 05 测试、评分与绘图

```bash
export OUTPUT_DIR=artifacts/ruler_qa/benchmarks/qa_main_v1
CHECKPOINT="$RUN_DIR/final" PLAN_ONLY=1 bash scripts/ruler_qa/sh/50_benchmark.sh
CHECKPOINT="$RUN_DIR/final" bash scripts/ruler_qa/sh/50_benchmark.sh
BENCHMARK_DIR="$OUTPUT_DIR" bash scripts/ruler_qa/sh/70_plot_heatmap.sh
```

默认在同一测试网格上按 `4 8 16 32` 四个固定压缩比分别贪心生成，每题最多 32 个新 token。用 `RATIOS='4 8'` 缩短评测，用 `TEST_DIR` 指定另一份已准备网格。任一压缩比需 ≥ 网格的 `min_ratio`，生成上限需 ≤ `generation_budget`。

设置 `MAX_TEST_SAMPLES=32`（Python 参数 `--max-test-samples 32`）可让每个压缩比最多评测 32 条。脚本按任务 × 长度格子轮流选取固定子集，所有压缩比复用同一批样本。默认评测全量。子集热力图将未采样的格子显示为 `NA`。

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2   CHECKPOINT="$RUN_DIR/final_merged"   OUTPUT_DIR=artifacts/ruler_qa/benchmarks/qa_merged_v1   bash scripts/ruler_qa/sh/50_benchmark.sh
```

多卡各加载完整模型对，并对样本分片。每个压缩比的所有分片齐全后才汇总。RULER QA 训练得到的 checkpoint 会额外检查训练问题哈希与测试题是否重叠。使用其他流程的 SeCo checkpoint 时，仍校验 tokenizer 和测试清单，训练数据重叠需由该实验自行核查。

| 指标 | 定义 |
| --- | --- |
| `ruler_score` | 复用 RULER v1 QA 的部分匹配定义。预测经上游风格的控制字符处理和首尾去空白后，包含任一参考答案的大小写无关子串即计 1 |
| `em` | 标准化后与任一参考答案完全一致计 1。标准化复用仓库的英文大小写、标点、冠词和空白规则 |
| `f1` | 标准化后计算 token 多重集重叠 F1，取所有参考答案中的最高值 |
| `count` / `empty_count` | 当前格子的预测数与空预测数 |
| `mean_seconds_per_sample` | CUDA 同步计时后的每样本均摊生成时间，含编码、压缩和解码，排除加载、拷贝和 warm-up |
| `peak_allocated_gib` | 各 GPU 的峰值 PyTorch CUDA allocated memory 最大值 |

子串指标对 HotpotQA 的 `yes` / `no` 短答案可能命中更长的单词，可同时看 EM 和 F1。JSON 与 CSV 的分数统一为 0–1；热力图用百分数。`summary` 是按所有测试行求均值，`tasks` 按任务分开，`cells` 按任务和长度分开。[RULER v1 QA 评分实现](https://github.com/NVIDIA/RULER/blob/main/scripts/eval/synthetic/constants.py)

| 输出 | 内容 |
| --- | --- |
| `benchmark.json` | checkpoint 身份、测试清单、生成设置和设备信息 |
| `ratio_<R>/predictions.jsonl` | 逐题 ID、问题、参考答案、预测、token 数、分数与耗时 |
| `ratio_<R>/metrics.json` | 该压缩比的总分、分任务及分长度分数、运行开销 |
| `ruler_qa_heatmap.png` / `.pdf` | 单图双面板。横轴是目标 token 预算，纵轴是压缩比，两面板共用 0–100 色标 |
| `ruler_qa_heatmap.csv` | 每个任务 × 长度 × 压缩比格子的三个指标和样本数 |

评分在评测时自动执行。修改评分实现后可用完整预测文件离线重算。绘制其他指标时建议显式设置图名。

```bash
PREDICTIONS="$OUTPUT_DIR/ratio_4/predictions.jsonl" bash scripts/ruler_qa/sh/60_score.sh
BENCHMARK_DIR="$OUTPUT_DIR" METRIC=f1 FIGURE_PATH="$OUTPUT_DIR/ruler_qa_f1.png"   bash scripts/ruler_qa/sh/70_plot_heatmap.sh
```
