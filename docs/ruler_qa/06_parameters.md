# 06 配置与脚本变量

相对路径从项目根目录解析。Shell 变量用 `变量=值 bash 脚本.sh` 设置，带空格的值用引号。各 `py` 入口支持 `--help`，例如 `python scripts/ruler_qa/py/seco/50_benchmark.py --help`。

## `configs/ruler_qa/prepare.json`

| 参数 | 默认 | 含义和修改提示 |
| --- | --- | --- |
| `train_lengths` | `[4096,8192]` | 训练总 token 桶，含答案 EOS。改变后重做训练数据 |
| `test_lengths` | `[4096,8192,16384,32768]` | 测试总 token 桶，含生成预留。改变后重做测试网格 |
| `train_candidates_per_task` | `4096` | 每任务流式随机采样后的候选上限，也决定训练干扰池规模 |
| `train_samples_per_task` | `512` | 每任务最终训练题数，应至少覆盖所有训练长度 |
| `test_samples_per_task` | `50` | 每任务每个长度的评测题数。总测试行数为 `2 × 题数 × 长度数` |
| `min_fill` | `0.85` | 实际总 token 数达到目标预算的最低比例，取值 `(0,1]` |
| `min_ratio` | `4` | 预处理使用的最低压缩比。训练或评测允许的比率不得低于对应清单 |
| `max_decoder_tokens` | `16384` | 解码器的总预算，计入压缩后的上下文、问题和答案或预留 |
| `max_answer_tokens` | `64` | 训练答案长度上限，含 EOS |
| `generation_budget` | `32` | 测试为生成预留的 token 数。`MAX_NEW_TOKENS` 需 ≤ 此值 |
| `seed` | `42` | 数据候选、干扰文档和顺序的种子。变更后建议另存数据目录 |

## `configs/ruler_qa/seco/train.json`

| 参数 | 默认 | 含义和修改提示 |
| --- | --- | --- |
| `model.name_or_path` / `revision` | `meta-llama/Llama-3.2-1B-Instruct` / `main` | 基座与 Hub revision。`MODEL_PATH` 可覆盖前者。用同一 tokenizer 预处理 |
| `model.dtype` / `attn_implementation` | `bfloat16` / `sdpa` | 权重精度和注意力实现。可选精度为 `bfloat16`、`float16`、`float32`。后端可选 `eager`、`sdpa`、`flash_attention_2`，最后一项需要额外安装 |
| `model.similarity_chunk_tokens` | `16` | 压缩阶段的相似度计算分块大小。更小的块降低临时内存，可能增加耗时 |
| `model.ratio_mode` / `compress_ratio` | `random` / `16` | 随机或固定模式。固定模式使用 `compress_ratio` |
| `model.random_ratios` | `[4,8,16,32]` | 随机模式从数组等概率抽取一个索引 |
| `model.lora_r` / `lora_alpha` / `lora_dropout` | `128` / `32` / `0.05` | 编码器和解码器 LoRA 秩、缩放和 dropout |
| `data.prepared_dir` | `artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/prepared/train_v1` | Python 的训练数据默认目录。Shell 用 `PREPARED_DIR` 传入 |
| `per_device_train_batch_size` / `gradient_accumulation_steps` | `1` / `8` | 单设备 microbatch / 每次优化累积步数 |
| `learning_rate` / `optim` | `1e-5` / `adamw_torch` | 初始学习率和优化器 |
| `num_train_epochs` / `max_steps` | `1` / `-1` | epoch 与累计步上限。正的 `max_steps` 优先 |
| `lr_scheduler_type` / `warmup_ratio` | `cosine` / `0.1` | 学习率调度和 warm-up 占总优化步数的比例 |
| `weight_decay` / `max_grad_norm` | `0.01` / `1.0` | 权重衰减与梯度裁剪阈值 |
| `gradient_checkpointing` | `true` | 重算激活，节省训练显存 |
| `save_strategy` / `save_steps` / `save_total_limit` | `steps` / `50` / `3` | 每 50 个优化步保存可续训检查点，最多保留 3 个。`final` 单独输出 |
| `logging_steps` / `report_to` | `10` / `none` | 训练日志间隔与外部报告集成 |
| `dataloader_num_workers` / `dataloader_pin_memory` | `2` / `true` | 加载 worker 数与 pinned memory |
| `seed` / `data_seed` | `42` / `42` | 模型训练 RNG 与训练集顺序种子 |
| `eval_strategy` | `no` | 测试由独立生成评测脚本执行 |

除 `model.*` 和 `data.*` 外，上表训练参数都位于 `training` 节。

## Shell 变量

| 脚本 | 变量 | 默认或作用 |
| --- | --- | --- |
| 通用 | `PYTHON_BIN`、`RULER_QA_ROOT` | `python`、`artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa`。后者控制数据、日志和实验目录的默认值 |
| 通用 | `HF_ENDPOINT`、`HF_HOME`、`OMP_NUM_THREADS` | Hub 地址、缓存目录、每进程默认 CPU 线程数 `4`。Shell 将 `TOKENIZERS_PARALLELISM` 设为 `false` |
| `00` | `TORCH_INDEX`、`INSTALL_DEEPSPEED` | CUDA wheel 索引和 ZeRO 依赖开关，详见 [环境](01_environment.md) |
| `01` | `MODEL_ID`、`MODEL_PATH`、`REVISION` | 下载的 Hub ID、本地输出目录、请求的 Hub revision |
| `10` | `SOURCES`、`RAW_DIR` | `configs/ruler_qa/sources.json`、`$RULER_QA_ROOT/raw` |
| `20` | `SPLIT`、`PREPARE_CONFIG` | `train`（或 `test`）、`configs/ruler_qa/prepare.json` |
| `20` | `PROMPT_STYLE` | `raw`（或 `legacy`）。对应 `--prompt-style`，详见 [03 数据预处理](03_preprocessing.md)。 |
| `20` | `RAW_DIR`、`PREPARED_DIR`、`MODEL_PATH`、`REVISION` | 原始数据目录、新输出目录、本地快照或 Hub ID、Hub revision。输出默认是 `prepared/train_v1` 或 `prepared/test_v1` |
| `30` | `CONFIG`、`PREPARED_DIR`、`MODEL_PATH` | 训练 JSON、训练数据目录、基座覆盖项。默认配置在本地快照存在时优先用本地快照 |
| `30` | `CUDA_VISIBLE_DEVICES`、`NPROC_PER_NODE` | 默认 `0` 和 `1`。每 GPU 一个进程，进程数需匹配可见设备 |
| `30` | `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` | 默认 `1`、`0`、`127.0.0.1`，端口按 SeCo、COMI、ICAE、RAM 使用 `29610–29613`。多节点需配置主节点地址及各节点序号，并共享 `RUN_DIR` 和数据 |
| `30` | `BACKEND`、`RUN_DIR` | 默认 `ddp` 和 UTC 时间戳加随机后缀目录 |
| `30` | `RESUME`、`MAX_STEPS`、`PLAN_ONLY` | 可续训检查点、累计步上限、预检开关 `0/1` |
| `40` | `CHECKPOINT`、`MERGED_DIR`、`MERGE_DEVICE`、`MODEL_PATH` | 输入适配器、合并输出（默认原路径加 `_merged`）、`cpu` 或 `cuda`、可选迁移后的同版本基座 |
| `50` | `CHECKPOINT`、`TEST_DIR`、`OUTPUT_DIR`、`MODEL_PATH` | 必填模型目录、测试网格、新评测目录、可选迁移后的同版本基座 |
| `50` | `EVAL_TRAIN`、`PREPARED_DIR` | 默认 `0`。设为 `1` 评测训练集，训练目录默认 `$RULER_QA_ROOT/prepared/train_v1`，详见 [08 训练集评测](08_train_benchmark.md) |
| `50` | `MAX_TEST_SAMPLES` | 默认留空使用全量。设置正整数可限制每个压缩比的总行数，适用于训练集和独立测试集 |
| `50` | `CUDA_VISIBLE_DEVICES`、`NPROC_PER_NODE` | 默认 `0` 和 `1`。测试脚本在单节点按 GPU 分片 |
| `50` | `RATIOS`、`BATCH_SIZE`、`WORKERS` | 空格分隔的 `4 8 16 32`、单卡 batch `1`、加载进程 `2` |
| `50` | `WARMUP_BATCHES`、`MAX_NEW_TOKENS`、`PLAN_ONLY` | 每卡每个压缩比预热 `1` 次、最多生成 `32` token、预检开关 `0/1` |
| `50` | `REPETITION_PENALTY`、`ATTENTION`、`NO_MERGE` | 默认重复惩罚 `1.0`，注意力后端沿用 checkpoint，默认合并 LoRA 后推理。设 `NO_MERGE=1` 保留 LoRA 模块 |
| `60` | `PREDICTIONS` | 必填，指向某一压缩比的完整 `predictions.jsonl` |
| `70` | `BENCHMARK_DIR`、`METRIC`、`FIGURE_PATH`、`DPI` | 必填评测目录，默认 `ruler_score`、目录下的 `ruler_qa_heatmap.png`、`300` DPI |

`50` 支持通过 Python 参数 `--attention` 覆盖 checkpoint 的注意力后端，或用 `--no-merge` 保留 LoRA 模块。`60` 支持 `--test-manifest` 指定独立清单，`--output` 指定新的指标路径。这些参数均可追加在相应 Shell 命令末尾。
