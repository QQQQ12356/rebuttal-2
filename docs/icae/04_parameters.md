# 04 ICAE 参数索引

常用变量可编辑编号 Bash 入口顶部，也可在命令前传入同名环境变量。传在 Bash 末尾的 CLI 参数交给 Python 解析。修改模型结构或训练配方时，复制 JSON 配置并通过 `CONFIG` 指定。

| 预处理变量 | Python 参数 | 默认与修改提示 |
| --- | --- | --- |
| `CORPUS_FILE` | `--input` | 单文件可用环境变量，多文件直接传 `--input file1 file2` |
| `TEXT_FIELD` | `--text-field` | `text`，填写实际文本列 |
| `MODEL_PATH` / `REVISION` | `--model` / `--revision` | 默认方法目录下的本地 Llama 快照和 `main` |
| `PREPARED_DIR` | `--output` | 默认 `icae_pretrain/prepared/v1`，目标目录需为新目录 |
| `MIN_TOKENS` / `MAX_TOKENS` | `--min-tokens` / `--max-tokens` | 512 / 4096，限制切分前的整篇文档长度 |
| `TRAIN_SAMPLES` / `EVAL_SAMPLES` | `--train-samples` / `--eval-samples` | 10000 / 1000，验证数量可设 0 |

| 训练变量 | Python 参数 | 默认与修改提示 |
| --- | --- | --- |
| `CONFIG` | `--config` | 预训练用 `configs/icae/pretrain.json`，微调用任务下的 `icae/train.json` |
| `PREPARED_DIR` | `--prepared-dir` | 指向当前阶段的清单和数据文件 |
| `MODEL_PATH` | `--model` | 覆盖 JSON 的基座路径，和分词时使用同一快照 |
| `INIT_FROM` | `--init-from` | 新建微调 run 时指向预训练适配器目录 |
| `RESUME` | `--resume` | 恢复当前阶段的 `checkpoint-N` |
| `RUN_DIR` | `--output` | 新训练自动命名，恢复默认使用 checkpoint 的父目录 |
| `MAX_STEPS` | `--max-steps` | 空值使用 JSON，正数设置累计 optimizer updates |
| `PLAN_ONLY` | `--plan-only` | 设为 1 时仅做预检 |
| `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE` | torchrun 环境及参数 | 预训练默认 `0` / `1`，每张可见 GPU 对应一个进程 |
| `MASTER_PORT` | torchrun `--master_port` | 预训练默认 29620。重复并行启动预训练时为各 run 分配独立端口。微调端口见[训练指南](../03_training.md#并行运行的默认端口) |
| `BACKEND` | `--backend` | 预训练默认 `ddp`，可选 `zero2` 和 `zero2_cpu` |
| `BACKBONE` / `ICAE_PRETRAIN_ROOT` | 环境变量 | 目录中的基座名和预训练产物根，替换基座时同时改实际模型路径 |
| `PYTHON_BIN` / `ARTIFACT_ROOT` | 环境变量 | Linux Python 解释器和实验总根目录 |

| JSON 字段 | 默认与修改提示 |
| --- | --- |
| `model.pretraining.ae_weight` / `lm_weight` | 预训练均为 1，非负且总和为正 |
| `model.memory_slots` | 32，改变会改变参数形状，两个阶段须一致 |
| `model.compress_ratio` / `ratio_mode` | 32 / fixed。每块原文容量为 `memory_slots × ratio`。研究随机比例时显式设置 `random_ratios` |
| `model.full_tail_slots` | true，尾块也保留全部槽。解码预算用 `ceil(context_tokens / chunk_size) × memory_slots + 1`，加上 prompt 和目标长度 |
| `model.keep_ae_marker` | 微调为 true，用于接收并保存预训练 AE 标记 |
| `data.require_pretrained` | 微调为 true，要求有预训练来源 |
| `model.lora_target_modules` / `lora_r` / `lora_alpha` / `lora_dropout` | `[q_proj, v_proj]` / 128 / 256 / 0。改动前三项需重做预训练或选择匹配的预训练权重 |
| `model.dtype` / `attn_implementation` | bfloat16 / sdpa，按 GPU 与运行环境调整 |
| `training.learning_rate` / `adam_beta2` / `weight_decay` | 预训练为 1e-4 / 0.95 / 0.1，微调以任务 JSON 为准 |
| `training.per_device_train_batch_size` / `gradient_accumulation_steps` | 1 / 8。改变进程数时同步检查有效 batch |
| `training.max_steps` / `num_train_epochs` | -1 / 1，正步数优先 |
| `training.gradient_checkpointing` | true，使用非重入激活重算 |
| `training.save_steps` / `save_total_limit` | 500 / 3，短实验如需验证恢复可将保存间隔设为 1 |
