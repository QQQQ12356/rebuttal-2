# 06 参数位置和修改提示

方法 Bash 入口顶部列出常用变量。高级参数在同任务的上一级共享 Bash 文件中列出，可在调用时通过环境变量覆盖，或把 Python CLI 参数追加在 Bash 命令末尾。直接调用方法的 Python 入口时，可编辑 `DEFAULT_ARGS` 元组或使用 `--help` 查看参数。实际传入的 CLI 参数在默认元组之后解析。模型和优化器参数集中在 `configs/<任务>/<方法>/train.json`。

| 通用变量 | 默认值／用途 |
| --- | --- |
| `PYTHON_BIN` | `python`，当前 Linux 环境中的解释器 |
| `BACKBONE` | `Llama-3.2-1B-Instruct`，产物目录的基座段 |
| `ARTIFACT_ROOT` | `artifacts`，所有实验的共同根目录 |
| `BENCHMARK_ROOT`、`NIAH_ROOT`、`RULER_QA_ROOT` | 分别覆盖任务产物目录 |
| `MODEL_ID`、`MODEL_PATH`、`REVISION` | Hub ID、本地快照目录和 Hub 版本 |
| `HF_HOME`、`HF_ENDPOINT` | 共用下载缓存和 Hub 服务地址 |
| `CONFIG` | 训练 JSON 路径。方法入口检查 JSON 的 `model.method` |
| `RUN_DIR`、`OUTPUT_DIR` | 训练和评测输出；重复实验用新目录 |
| `CHECKPOINT`、`MERGED_DIR`、`MERGE_DEVICE` | 待加载的权重目录、合并输出和执行设备 |
| `SAMPLES`、`MAX_TEST_SAMPLES` | Benchmark 每数据集上限与 NIAH/RULER QA 全局上限 |
| `WORKERS`、`WARMUP_BATCHES` | 数据加载进程数、各评测分片的预热批次数 |
| `METRIC`、`COLUMNS`、`DPI` | 绘图指标、NIAH 面板列数和分辨率 |

| JSON 字段 | 适用方法 | 默认值和作用 |
| --- | --- | --- |
| `model.name_or_path`、`revision` | 全部 | Llama 3.2 1B Instruct、main。脚本可传入本地快照 |
| `dtype`、`attn_implementation` | 全部 | bfloat16、sdpa。相同设置有利于比较显存和时延 |
| `lora_r`、`lora_alpha`、`lora_dropout` | 全部 | SeCo 为 128 / 32 / 0.05，COMI 和 RAM 为 32 / 32 / 0.05，ICAE 为 128 / 256 / 0 |
| `ratio_mode`、`compress_ratio`、`random_ratios` | 全部 | ICAE 默认 fixed / 32，其余默认 random / 16 / [4, 8, 16, 32]。训练每次 forward 从数组抽一个比例，fixed 时使用 compress_ratio |
| `similarity_chunk_tokens` | SeCo | 16，余弦计算分块大小。设 0 使用整块计算 |
| `encoder_layers` | COMI / RAM | 8，取 backbone 前若干层。上限为基座层数 |
| `fusion_layers` | COMI / RAM | 1，融合 Transformer 层数。0 跳过 Transformer 融合 |
| `segment_size` | COMI | 4096，上下文分段长度。段内含 query 编码，段边界各自做比例取整 |
| `memory_slots` / `full_tail_slots` | ICAE | 32 / true，chunk 容量为槽数 × 比例，默认尾块也使用完整槽数。见 [ICAE 参数](icae/04_parameters.md) |
| `segment_batch_size` | RAM | 256，每次编码的短段数。降低可减少单次 encoder 激活峰值 |
| `keep_ratio` | RAM | 0.05，按 query 相关度保留的段比例，范围 [0, 1) |
| `contrastive_loss_weight` | RAM | 1.0，命中训练答案片段时的对比损失权重，设为 0 关闭 |
| `contrastive_temperature` | RAM | 0.1，对比分数温度 |
| `training.max_steps`、`num_train_epochs` | 全部 | 正步数优先。`max_steps=-1` 时按 epoch 训练 |
| `per_device_train_batch_size`、`gradient_accumulation_steps` | 全部 | 每 rank microbatch 与累积步数。多卡时同时考虑总有效 batch |
| `learning_rate`、`warmup_ratio`、`lr_scheduler_type` | 全部 | 优化学习率、预热占比与调度方式 |
| `gradient_checkpointing`、`save_steps`、`save_total_limit` | 全部 | 激活重算开关、周期保存步数、历史 checkpoint 保留上限 |

各任务的完整变量索引见 [Benchmark](benchmark/08_script_variables.md)、[NIAH](niah/07_parameters.md) 和 [RULER QA](ruler_qa/06_parameters.md)。COMI、ICAE 和 RAM 的压缩比例使用正整数。更换 backbone、模型结构或 LoRA 形状时，建立新训练目录并同步更新评测使用的 checkpoint。
