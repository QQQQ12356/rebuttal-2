# 04 ICAE 参数索引

常用变量可编辑编号 Bash 入口顶部，也可在命令前传入同名环境变量。传在 Bash 末尾的 CLI 参数交给 Python 解析。修改模型结构或训练配方时，复制 JSON 配置并通过 `CONFIG` 指定。

## 预训练数据脚本

| 预处理变量 | Python 参数 | 默认与修改提示 |
| --- | --- | --- |
| `CORPUS_FILE` | `--input` | 单文件可用环境变量，多文件直接传 `--input file1 file2` |
| `TEXT_FIELD` | `--text-field` | `text`，填写实际文本列 |
| `MODEL_PATH` / `REVISION` | `--model` / `--revision` | 默认方法目录下的本地 Llama 快照和 `main` |
| `PREPARED_DIR` | `--output` | 默认 `icae_pretrain/prepared/v1`，目标目录需为新目录 |
| `MIN_TOKENS` / `MAX_TOKENS` | `--min-tokens` / `--max-tokens` | 512 / 4096，限制切分前的整篇文档长度 |
| `TRAIN_SAMPLES` / `EVAL_SAMPLES` | `--train-samples` / `--eval-samples` | 10000 / 1000，验证数量可设 0 |

## 训练脚本

| 训练变量 | Python 参数 | 默认与修改提示 |
| --- | --- | --- |
| `CONFIG` | `--config` | 预训练用 `configs/icae/pretrain.json`，微调用任务下的 `icae/train.json` |
| `PREPARED_DIR` | `--prepared-dir` | 指向当前阶段的清单和数据文件 |
| `MODEL_PATH` | `--model` | 覆盖 JSON 的基座路径，和分词时使用同一快照 |
| `INIT_FROM` | `--init-from` | 默认空。使用预训练适配器时，填写预训练的 `final/` 或 `checkpoint-N/` |
| `INIT_FROM_BASE` | `--init-from-base` | 默认 0。设为 1 时用 `MODEL_PATH` 加载原始基座并新建 ICAE 参数 |
| `RESUME` | `--resume` | 恢复当前阶段的 `checkpoint-N` |
| `RUN_DIR` | `--output` | 新训练自动命名，恢复默认使用 checkpoint 的父目录 |
| `MAX_STEPS` | `--max-steps` | 空值使用 JSON，正数设置累计 optimizer updates |
| `PLAN_ONLY` | `--plan-only` | 设为 1 时仅做预检 |
| `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE` | torchrun 环境及参数 | 预训练和 RULER QA 默认 `0` / `1`。Benchmark 和 NIAH 默认 `0,1,2,3` / `4`。按可见 GPU 数设置进程数 |
| `MASTER_PORT` | torchrun `--master_port` | 预训练、Benchmark、NIAH、RULER QA 分别为 29620、29502、29602、29612。同一主机并行运行时使用独立端口 |
| `NNODES` / `NODE_RANK` / `MASTER_ADDR` | torchrun 多节点参数 | 微调默认 `1` / `0` / `127.0.0.1`。多节点时使用各自的 rank、rank 0 的可达地址和同一个共享 `RUN_DIR`。预训练脚本使用单机 |
| `RUN_NAME` | 运行目录名前缀 | Benchmark 和 NIAH 默认分别为 `benchmark` 和 `niah`。可用 `RUN_DIR` 指定完整路径 |
| `BACKEND` | `--backend` | 可选 `ddp`、`zero2`、`zero2_cpu`。预训练和 RULER QA 默认 `ddp`，Benchmark 和 NIAH 默认 `zero2` |
| `BACKBONE` / `ICAE_PRETRAIN_ROOT` | 环境变量 | 目录中的基座名和预训练产物根，替换基座时同时改实际模型路径 |
| `PYTHON_BIN` / `ARTIFACT_ROOT` | 环境变量 | Linux Python 解释器和实验总根目录 |

新建微调时在 `INIT_FROM` 和 `INIT_FROM_BASE=1` 中选择其一。续训时只设置 `RESUME`，Shell 会从 checkpoint 推导 `RUN_DIR`。直接传 `--resume` 时同时指定 `RUN_DIR` 或 `--output`。

## JSON 配置

| JSON 字段 | 默认与修改提示 |
| --- | --- |
| `model.pretraining.ae_weight` / `lm_weight` | 预训练均为 1，非负且总和为正 |
| `model.memory_slots` | 32，改变会改变参数形状，两个阶段须一致 |
| `model.compress_ratio` / `ratio_mode` | 32 / fixed。每块原文容量为 `memory_slots × ratio`。研究随机比例时显式设置 `random_ratios` |
| `model.full_tail_slots` | true，尾块也保留全部槽。解码预算用 `ceil(context_tokens / chunk_size) × memory_slots + 1`，加上 prompt 和目标长度 |
| `model.keep_ae_marker` | 微调为 true，用于接收并保存预训练 AE 标记 |
| `data.require_pretrained` | 微调模板为 true。默认要求显式选择预训练适配器、基座初始化或续训。自定义配置设为 false 时，省略初始化选项将直接从基座开始 |
| `model.lora_target_modules` / `lora_r` / `lora_alpha` / `lora_dropout` | `[q_proj, v_proj]` / 128 / 256 / 0。改动前三项需重做预训练或选择匹配的预训练权重 |
| `model.dtype` / `attn_implementation` | bfloat16 / sdpa，按 GPU 与运行环境调整 |
| `training.learning_rate` / `adam_beta2` / `weight_decay` | 预训练为 1e-4 / 0.95 / 0.1，微调以任务 JSON 为准 |
| `training.per_device_train_batch_size` / `gradient_accumulation_steps` | 1 / 8。改变进程数时同步检查有效 batch |
| `training.max_steps` / `num_train_epochs` | -1 / 1，正步数优先 |
| `training.gradient_checkpointing` | true，使用非重入激活重算 |
| `training.save_steps` / `save_total_limit` | 500 / 3，短实验如需验证恢复可将保存间隔设为 1 |
| `model.name_or_path` / `revision` | 基座来源和版本。使用 Hub 时可固定 commit，使用本地模型时让训练与预处理指向同一快照 |
| `training.optim` / `max_grad_norm` | `adamw_torch` / 1，分别控制优化器和梯度裁剪 |
| `training.lr_scheduler_type` / `warmup_ratio` | `cosine` / 0.1，分别控制学习率曲线和预热步数比例 |
| `training.save_strategy` / `eval_strategy` / `logging_steps` | `steps` / `no` / 10。周期保存、默认关闭训练中评估、每 10 步记录一次 |
| `training.dataloader_num_workers` / `dataloader_pin_memory` | 2 / true，控制数据加载子进程和页锁定内存 |
| `training.seed` / `data_seed` / `report_to` | 42 / 42 / `none`，分别控制初始化、数据顺序和外部日志上报 |
