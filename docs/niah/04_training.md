# 04 NIAH 训练与断点恢复

NIAH 使用自己的 `configs/niah/train.json`、入口和 run 目录。数据 manifest 必须是 NIAH train 格式；原五数据集训练入口会拒绝 NIAH 数据，NIAH 训练入口也拒绝旧 QA 数据。共享实现集中在 `seco/train_cli.py`，不复制模型/优化器/保存代码。

## 1. 启动命令

```bash
# 只验证配置、数据哈希、比例预算和输出计划，不加载模型。
PLAN_ONLY=1 bash scripts/niah/sh/30_train.sh

# 默认四卡 ZeRO-2。
bash scripts/niah/sh/30_train.sh

# 单卡或两卡 DDP。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp bash scripts/niah/sh/30_train.sh
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BACKEND=ddp bash scripts/niah/sh/30_train.sh

# 卸载方案。
BACKEND=zero2_cpu bash scripts/niah/sh/30_train.sh
BACKEND=zero3_cpu bash scripts/niah/sh/30_train.sh
```

| backend | 保存/显存行为 | 使用提示 |
| --- | --- | --- |
| ddp | 每卡完整双模型与优化器 | 最先验证的基准路径 |
| zero2 | 优化器/梯度分片，基座仍每卡完整 | 默认多卡方案 |
| zero2_cpu | 优化器状态 CPU 卸载，基座仍每卡完整 | LoRA 优化器占比小，可能省显存有限 |
| zero3_cpu | 参数分片/按需 CPU 搬运，优化器 CPU 卸载 | 权重本身放不下时考虑，需 RAM 与通信带宽 |

这些 backend 复用根 `configs/zero*.json`。如果只想调整 NIAH 的 ZeRO 配置而保留 QA 配置，先复制配置并在共享映射中增加明确新 backend；不要不加区分地修改公共 JSON。没有启用 tensor parallel 或 DeepSpeed inference。

有效 batch=`WORLD_SIZE×per_device_train_batch_size×gradient_accumulation_steps`，默认四卡为 `4×1×8=32`。模型仍是独立 encoder/decoder，各自 LoRA；训练不使用 device_map=auto。

## 2. 比例模式与模型字段

| model 字段 | 默认 | 含义 |
| --- | --- | --- |
| name_or_path / revision | Qwen/Qwen3-4B-Instruct-2507 / main | 默认基座与远程版本；Shell 会传入本地 MODEL_PATH |
| dtype | bfloat16 | 按硬件选择 bfloat16/float16/float32，保持对照一致 |
| attn_implementation | sdpa | 可选 eager、flash_attention_2；后者需要匹配安装 |
| similarity_chunk_tokens | 16 | 原公式余弦广播分块大小；0 用原始整块路径 |
| ratio_mode | random | random 或 fixed |
| random_ratios | [4,8,16,32] | 每次训练 forward 抽一个，同 microbatch 共用；重复值相当于加权 |
| compress_ratio | 16 | fixed 模式使用的比例；测试由 --ratios 明确覆盖 |
| lora_r / lora_alpha / lora_dropout | 128 / 32 / 0.05 | 两套分支的 LoRA 设置，不能在加载 checkpoint 时随意改 |
| data.prepared_dir | artifacts/niah/prepared/train_v1 | 独立训练产物目录；Shell PREPARED_DIR 覆盖 |

随机模式使用可被 native Trainer checkpoint 恢复的 torch CPU RNG。改为 fixed 只需设置 `ratio_mode=fixed` 并设定 compress_ratio；random_ratios 可保留为未启用配置。若训练比例小于数据预处理 limits.min_ratio，入口会报错；降低比例时重新预处理，不仅修改训练 JSON。

## 3. training 字段

| 字段 | 默认 | 用途 |
| --- | --- | --- |
| per_device_train_batch_size / gradient_accumulation_steps | 1 / 8 | microbatch 与有效 batch |
| learning_rate / weight_decay | 1e-5 / 0.01 | LoRA 优化设置 |
| num_train_epochs / max_steps | 1 / -1 | max_steps>0 时以累计 optimizer steps 为准 |
| lr_scheduler_type / warmup_ratio | cosine / 0.1 | 学习率调度 |
| max_grad_norm | 1 | 梯度裁剪 |
| gradient_checkpointing | true | 双模型非重入激活重算 |
| save_strategy / save_steps / save_total_limit | steps / 500 / 3 | 周期 native checkpoint 与保留数量 |
| logging_steps | 10 | optimizer-step 日志间隔 |
| dataloader_num_workers / dataloader_pin_memory | 2 / true | 加载进程与 pinned memory |
| seed / data_seed | 42 / 42 | 训练与数据 shuffle RNG |
| report_to | none | 不默认上报在线服务 |
| optim | adamw_torch | DDP/普通 ZeRO-2 优化器；CPU 卸载配置定义 DeepSpeed AdamW |
| eval_strategy | no | 不在训练循环使用 NIAH 测试网格选模型 |

输出目录、DeepSpeed 配置、bf16/fp16、列保留和保存策略的必要控制项由 launcher 注入。不要在 JSON 重复添加 output_dir/deepspeed/bf16/fp16/remove_unused_columns/save_only_model；未提供训练期验证集，不启用 load_best_model_at_end。

## 4. 保存、恢复与合并

默认 run 位于 `artifacts/niah/runs/niah_qwen3_4b_<UTC时间>_<随机后缀>/`，包括 experiment.json、data_manifest.json、checkpoint-N、final。checkpoint-N 保留 Trainer/原生 DeepSpeed optimizer、scheduler、RNG 状态；final 是推理导出，不作为训练恢复点。小 safetensors 保存双 adapter，推理需要相同基座；ZeRO 原生 checkpoint 可能很大，不能只复制小权重文件恢复。

```bash
RESUME=artifacts/niah/runs/真实run/checkpoint-500 MAX_STEPS=2000 \
    bash scripts/niah/sh/30_train.sh
```

恢复要求同模型、数据 manifest、backend、world size 和配置，只允许变更累计 max_steps。run 目录默认由 RESUME 的父目录确定。改变比例数组、样本集、LoRA 或进程数应开新 run。

训练归档还保存 NIAH 背景文档与已选答案的哈希；NIAH benchmark 会再与当前 checkpoint 核对背景/答案重复。不同训练实验应共用同一份固定 held-out grid，不能因为某次结果低而换测试 seed。

合并使用独立 `scripts/niah/sh/40_merge_lora.sh`，底层复用统一 PEFT 合并实现。默认 CPU 合并、输出 `${CHECKPOINT}_merged`；已有目录拒绝覆盖。完整多节点设置及 native checkpoint 的限制与 [原训练指南](../04_training.md) 一致。
