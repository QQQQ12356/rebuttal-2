# 05 模型合并、推理与 EM/F1

贪心、采样、beam search 及脚本变量的完整说明见 [01 评测解码配置](../01_decoding.md)。

## 1. 合并并持久保存

```bash
CHECKPOINT=artifacts/runs/真实run/final bash scripts/benchmark/sh/40_merge_lora.sh
```

输出默认 `final_merged/`，内部有 HF 格式的 `encoder/`、`decoder/`、`tokenizer/`、`artifact.json` 和 `experiment.json`。使用 PEFT `merge_and_unload(safe_merge=True)`，不手写 LoRA 权重 key 或 alpha/r。合并输出完成后才发布正式目录，已有目录拒绝覆盖。默认 CPU 合并，需要容纳两套模型与合并临时张量的 RAM；可用 `MERGE_DEVICE=cuda` 切换。

也可以直接测试 `final/` 或 `checkpoint-N/`：程序加载基座+两套 adapter，并在内存合并。重复评测优先使用持久合并目录，省去重复 adapter 初始化/合并。`--no-merge` 可在 adapter checkpoint 上保留 LoRA 模块，用于小样本合并前后比较；对已经合并的目录无法撤销合并。

## 2. 运行四数据集 benchmark

```bash
CHECKPOINT=artifacts/runs/真实run/final_merged bash scripts/benchmark/sh/50_benchmark.sh

# 四卡数据并行，每卡一份完整 encoder/decoder。
CHECKPOINT=artifacts/runs/真实run/final_merged \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
INCLUDE_CONTEXT=1 bash scripts/benchmark/sh/50_benchmark.sh

# 只测试 NaturalQA、HotpotQA 和两种比例。
python -m torch.distributed.run --standalone --nproc_per_node 1 scripts/benchmark/py/50_benchmark.py \
    --checkpoint artifacts/runs/真实run/final_merged \
    --output artifacts/benchmarks/custom_test \
    --datasets naturalqa hotpotqa --ratios 8 16
```

程序每进程只加载一次模型，然后顺序测试所有比例。先固定数据选择，rank 使用不重叠步长分片，不用补齐式 DistributedSampler；每个 rank 内按数据集/长度排序减少 padding；主进程检查 ID 集合与数量，再恢复原始选样顺序。空分片也会写空分片文件并参与同步。每个比例使用同一份样本，预热重复首 batch，不消耗或漏评测试样本。

启动脚本会打印日志路径，可用 `tail -f` 跟踪。进度按 GPU rank、压缩比和数据集显示已完成样本数、百分比和耗时。首批与末批结束时立即记录，中间批次在距上次输出至少 10 秒时更新。

推理多卡是提高吞吐的数据并行，**每张卡仍要放下一套完整双模型**，不提供 tensor parallel 或 ZeRO inference。训练能靠 ZeRO-3 放下，不意味着单卡推理也能放下。遇到 OOM 应缩短输入/降低 batch，或选用更大显存 GPU；不要将训练卡数当作推理模型分片数。

## 3. 参数

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| --checkpoint | 必填 | 新格式 final/checkpoint-N 或 merged 目录 |
| --model | 无 | adapter checkpoint 基座迁移后的路径；权重身份必须与训练一致 |
| --prepared-dir | artifacts/prepared/default | 已预处理四个测试源 |
| --output | 必填 | 新 benchmark 输出目录 |
| --datasets | 四个 benchmark | naturalqa / 2wikimqa / hotpotqa / multinews |
| --ratios | 4 8 16 32 | 固定评测比例数组，不能包含重复项或小于预处理 min_ratio 的值 |
| --batch-size | 1 | 每卡 batch；支持变长与尾部不足 batch |
| --samples | 0 | 每个已预处理测试集再取前 N；0 使用全部 prepared 样本 |
| --shuffle / --seed | 关闭 / 42 | 可选固定种子重排后再取 N |
| --workers | 2 | 每卡 DataLoader worker 数 |
| --warmup-batches | 1 | 每数据集/比例重复首 batch 的预热次数；0 关闭 |
| --qa-max-new-tokens | 64 | 三个 QA 测试源生成上限 |
| --news-max-new-tokens | 512 | MultiNews 摘要生成上限 |
| --repetition-penalty | 1.0 | 重复惩罚；1.0 不惩罚；对照实验统一 |
| --attention | 训练配置值 | 可切 eager/sdpa/flash_attention_2 做性能验证 |
| --no-merge | 关闭 | adapter 原样运行；用于合并前后等价性检查 |
| --include-context | 关闭 | prediction 文件保存完整 context，方便导入 Excel；增大磁盘占用 |

默认使用贪心解码并启用 KV cache。生成长度超过 preprocessing generation budget 时拒绝运行，避免预算不一致。不同任务分开配置 max_new_tokens，不把 `test_fast` 的固定 10-token 限制带入摘要任务。

## 4. 产物与指标

```text
artifacts/benchmarks/<benchmark>/
  benchmark.json                   模型/数据身份、参数、GPU 与计时说明
  ratio_4/
    predictions.rank0.jsonl        每 rank 独立结果
    runtime.rank0.json             每 rank 峰值 allocated 显存
    predictions.jsonl              合并完整结果，包含每样本 EM/F1
    metrics.json                   分数据集、micro、macro 指标
  ratio_8/ ...
```

EM 和 F1 均采用英文 SQuAD 式归一化：小写、去 ASCII 标点、去 a/an/the、合并空白；F1 用 token 多重集合重叠，保留重复 token 计数。多参考答案分别取最高 EM/F1。`micro` 按全部样本加权，`macro` 为本次参与的数据集等权平均。JSON 是 **0–1**，Excel 显示百分数。空预测对非空答案得零分；空归一化字符串对相同空字符串得 1。

MultiNews 也按要求只报告 EM/F1；这是与 QA 一致的文本匹配指标，不能将其直接解释成摘要内容质量。不要把宏平均与样本加权平均混用，也不要把少量/长度过滤后的测试结果当作完整官方 split 结果。

`seconds_per_sample` 是 GPU 同步后、encoder+压缩+decoder 的 batch 总秒数除以实际 batch 大小，排除模型加载、分词和 H2D。`mean_seconds_per_sample` 是该摊销值的均值；它不等于独立请求延迟或多卡墙钟吞吐。显存为各 rank 当前进程 allocated 峰值最大值，含该轮预热；不等于 nvidia-smi 总占用。比较加速需固定数据、比例、batch、生成长度、attention、精度、GPU 和预热条件。

## 5. 独立重新打分

```bash
PREDICTIONS=artifacts/benchmarks/真实benchmark/ratio_16/predictions.jsonl \
bash scripts/benchmark/sh/60_score.sh
```

默认重新生成同目录 `metrics.json` 并保留已有 ratio、计时与模型附注。`--output` 可指定新指标路径，`--scored-output` 可另存带重算分数的新 JSONL。新格式需要 `id/dataset/prediction/answer`；也兼容旧 SeCo 的 `subset/model_output/ground_truth`，缺 ID 时按行号生成。拒绝重复 ID 和空文件，不将缺失输出算成成功样本。

## 6. 原始 checkpoint 的迁移

原始 checkpoint 是整套双 LoRA 模型 state_dict，不能无校验地传给新 adapter loader。先在新的 JSON 配置中填写原始模型、原始 LoRA r/alpha/dropout，再导入：

```bash
# 必须改成原 checkpoint 的真实基座，不能拿默认 Qwen 加载原 Llama 权重。
python scripts/benchmark/py/41_import_checkpoint.py \
    --source /旧实验/model.safetensors --config configs/benchmark/原模型配置.json \
    --model /旧实验对应的原始基座 --output artifacts/imported/original_seco
CHECKPOINT=artifacts/imported/original_seco bash scripts/benchmark/sh/40_merge_lora.sh
```

导入检查全部 LoRA key、形状以及文件中存在的 frozen base 权重。基座内容不一致时拒绝丢弃自定义基座。只支持原始**未合并**双 LoRA safetensors；旧版手工合并文件不冒充新格式。导入用于推理，不恢复旧优化器、scheduler 或训练步数。旧训练没有可验证的数据来源时不补造来源元数据。
