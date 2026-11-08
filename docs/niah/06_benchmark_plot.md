# 06 NIAH 测试、指标与热力图

## 1. 生成预测

先填写真实 checkpoint。可以使用 NIAH 训练后的 final/checkpoint-N、merged 模型，也可以使用本工程已有兼容 SeCo checkpoint 作为未做 NIAH 训练的对照。

```bash
CHECKPOINT=artifacts/niah/runs/真实run/final bash scripts/niah/sh/40_merge_lora.sh

CHECKPOINT=artifacts/niah/runs/真实run/final_merged \
PLAN_ONLY=1 bash scripts/niah/sh/50_benchmark.sh

CHECKPOINT=artifacts/niah/runs/真实run/final_merged \
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
bash scripts/niah/sh/50_benchmark.sh
```

plan-only 检查 checkpoint 元数据、grid 文件哈希、比例和生成预算，打印样本数/最长输入/调用规模，不加载 torch 或模型；模型真实位置上限在实际加载后检查。NIAH checkpoint 保存的训练正文/答案哈希会再与本 grid 核对。

每个比例使用同一固定网格。rank 按不重叠索引分片，每卡独立生成，支持尾部小 batch 与空 shard，不为补齐 sampler 重复样本。每 rank 首 batch 可预热，但全部样本只计分一次。结束时主进程检查完整 grid ID、长度/深度/trial 集合后写指标。

| Python 参数 | 默认 | 说明 |
| --- | --- | --- |
| --checkpoint | 必填 | adapter final/checkpoint-N 或完整 merged 目录 |
| --model | 无 | adapter 的原始基座迁移路径，必须保持相同权重 |
| --grid-dir | artifacts/niah/prepared/grid_v1 | 独立固定测试网格 |
| --output | 必填 | 新 NIAH benchmark 目录，拒绝覆盖 |
| --ratios | 4 8 16 32 | 各压缩比使用相同样本，无重复项且不小于 grid.min_ratio |
| --max-test-samples | 全部 | 每个压缩比的样本总数上限，须为正整数 |
| --batch-size / --workers | 1 / 2 | 每卡 batch 与数据加载 worker |
| --warmup-batches | 1 | 重复首 batch 预热次数，0 关闭 |
| --max-new-tokens | 64 | 不得超过 grid.generation_budget |
| --repetition-penalty | 1.0 | greedy 解码的重复惩罚；对照统一 |
| --attention | checkpoint 值 | 可选 eager/sdpa/flash_attention_2 |
| --no-merge | 关闭 | adapter 原样运行，用于合并前后比较 |
| --plan-only | 关闭 | 不分配 CUDA 的计划检查 |

设置 `MAX_TEST_SAMPLES=32` 可让每个压缩比最多评测 32 条。脚本按长度 × 深度格子轮流选取固定子集，所有压缩比复用同一批样本。子集指标会标记为不完整网格，绘图时传入 `--allow-incomplete`。

多卡测试为数据并行，每张卡仍需容纳完整双模型。训练 ZeRO-3 能放下不代表测试也能放下；长注意力和原始 SeCo 聚类仍可能耗时较长，先验证显存。

输出：

```text
artifacts/niah/benchmarks/<独立benchmark>/
  benchmark.json                 参数、checkpoint/grid 身份、GPU、计时定义
  ratio_4/
    predictions.rankN.jsonl      每 batch 刷盘的分片结果
    runtime.rankN.json           当前进程峰值 allocated 显存
    predictions.jsonl            完整合并，含真实位置、答案、预测、分数
    metrics.json                 全局与逐格指标
  ratio_8/ ...
```

无自动 benchmark 续测；失败后保留分片用于定位，换新 OUTPUT_DIR 重跑。没有完整预测时不会发布成功的完整指标。若需要观察已经完成的部分，可显式重新计分允许不完整网格，图中会标注，不能当作完整 benchmark 成绩。

## 2. 计分定义

| 字段 | 定义 | 使用方式 |
| --- | --- | --- |
| accuracy / em | 归一化 exact match，正确=1、错误=0；每格取 trial 平均 | 默认热力图准确度 |
| f1 | 与原工程一致的 token 多重集合 F1，多参考答案取最高 | 辅助部分匹配指标 |
| hit_rate | 输出中包含完整目标答案，大小写忽略，并检查英文数字边界 | 可选宽松诊断，不能替代默认 EM |

归一化 EM/F1 使用现有 `seco.metrics`：英文小写、去 ASCII 标点、去 a/an/the、合并空白。不会只截第一行或删除模型解释后再评估。UUID 的 EM 是整个归一化值匹配；passkey 保留前导零，不转整数。hit_rate 不接受把 `12345678` 错匹配为 `9123456789` 的子串。

JSON 和 CSV 保存0–1，图显示0–100%。每格5次的 EM=0.8 表示4/5正确。summary 是全部已评分网格样本均值；固定 repeats 且完整时也等于格子等权均值。

参考 [Activation Beacon eval_needle.py](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/Long_LLM/activation_beacon/main/eval_needle.py) 提供 ROUGE-L recall，以及可选在线 GPT 1–10 分；用户 NIAH.png 的色条是后者。**本项目沿用长度/位置布局与红黄绿配色，但默认采用可复算的 EM 成功率，不把 EM 乘10称为 GPT 分数，也不伪造参考图成绩。**

计时为 batch encoder+压缩+decoder GPU 同步总秒数除以本 batch 实际样本数，排除加载、H2D 与数据构建；不是独立请求延迟或多卡墙钟吞吐。显存为本比例各 rank allocated 峰值的最大值，包含预热。

## 3. 重新打分与缺失格子

```bash
PREDICTIONS=artifacts/niah/benchmarks/真实benchmark/ratio_16/predictions.jsonl \
    bash scripts/niah/sh/60_score.sh

python scripts/niah/py/60_score.py \
    --predictions /路径/partial_predictions.jsonl \
    --grid-manifest artifacts/niah/prepared/grid_v1/manifest.json \
    --output /新路径/partial_metrics.json --allow-incomplete
```

默认从 prediction 上两级的 benchmark.json 读取嵌入 manifest；外部/部分文件需显式给 --grid-manifest。重复 ID、重复 cell/trial、非法 trial、失败 status、混合比例和空文件均拒绝。默认漏测也拒绝；allow-incomplete 时缺失格为 null，保留 count/expected_count，不能自动补零或补满。

## 4. 一张图展示多个比例

```bash
BENCHMARK_DIR=artifacts/niah/benchmarks/真实benchmark \
    bash scripts/niah/sh/70_plot_heatmap.sh

# 单个比例，接近用户参考图的单面板布局。
python scripts/niah/py/70_plot_heatmap.py \
    --inputs 'artifacts/niah/benchmarks/真实benchmark/ratio_16/metrics.json' \
    --output artifacts/niah/figures/R16.png --columns 1
```

| 绘图参数 | 默认 | 说明 |
| --- | --- | --- |
| --inputs | 必填 | 一个/多个 metrics 路径或加引号的 glob；按数值比例排序 |
| --output | 必填 | 全新 .png 路径；相同 stem 同时输出 PDF 和 CSV |
| --metric | accuracy | accuracy/em/f1/hit_rate；固定0–100色标，不按当前最小值缩放 |
| --columns | 2 | 多面板列数，一面板对应一个比例 |
| --title | SeCo Needle-in-a-Haystack | 总标题 |
| --dpi | 300 | PNG 分辨率 |
| --annotate-all | 关闭 | 默认隐藏100%文字，其他值显示；开启全部标数 |
| --allow-incomplete | 关闭 | 缺失灰色、标 NA，部分格显示 n/expected，标题标 INCOMPLETE |
| --no-pdf | 关闭 | 只输出 PNG/CSV |

横轴为精确序列 token 长度，纵轴0%在顶部、100%在底部。格子用离散等宽列，与参考图一致；不是按长度差距绘制连续面积。多个面板须使用相同任务、背景、seed、axes、repeats 与实际 grid 身份，防止拼接不同样本成绩。

输出 `NIAH.png` 可直接展示，`NIAH.pdf` 为可缩放图，`NIAH.csv` 包含各比例、长度、深度、数量和全部分数，支持进一步统计。PNG/PDF/CSV 任一已存在时拒绝覆盖，换 FIGURE_PATH。当前尚无目标 Linux 上的真实成绩，所以仓库不附带虚构的最终热力图。
