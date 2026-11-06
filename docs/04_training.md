# 04 训练、多卡与恢复

## 1. 先确认计划

```bash
PLAN_ONLY=1 bash scripts/sh/30_train.sh
```

计划检查配置、prepared manifest、训练 JSONL 哈希、压缩比预算和输出路径，然后打印配置与有效 batch。不加载模型、不分配 CUDA、不创建训练目录。实际 tokenizer 一致性和模型位置上限在训练初始化后再检查。

配置优先级为：Python 命令行覆盖项 > Shell 显式传入项 > `configs/train.json`。脚本默认向 Python 传入本地 `MODEL_PATH` 和 `PREPARED_DIR`，所以只修改 JSON 中的模型/数据路径还不够；也应改 Shell 变量，或直接运行 Python。

## 2. 启动方式

```bash
# 默认四卡 ZeRO-2。
bash scripts/sh/30_train.sh

# 单卡，不启用 DeepSpeed。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp bash scripts/sh/30_train.sh

# 两卡普通 DDP。
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BACKEND=ddp bash scripts/sh/30_train.sh

# 四卡，优化器状态 CPU 卸载。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 BACKEND=zero2_cpu bash scripts/sh/30_train.sh

# 参数、优化器状态 CPU 卸载。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 BACKEND=zero3_cpu bash scripts/sh/30_train.sh
```

| backend | 配置文件 | GPU 上保留内容 | 适用提示 |
| --- | --- | --- | --- |
| ddp | 无 DeepSpeed 配置 | 每卡完整参数与优化器状态 | 单卡或基准验证，最易排错 |
| zero2 | configs/zero2.json | 完整参数；优化器/梯度分片 | 多卡默认，权重本身仍需放下 |
| zero2_cpu | configs/zero2_cpu.json | 完整参数；优化器 CPU 卸载 | LoRA 优化器占比较小时收益有限 |
| zero3_cpu | configs/zero3_cpu.json | 参数按需搬运/分片；优化器 CPU 卸载 | 显存更紧张且 RAM/PCIe 可承受时使用 |

`TrainingArguments` 在模型构建之前创建，使 ZeRO-3 能在 `from_pretrained` 阶段分片基座。训练不使用 `device_map=auto`。ZeRO-3 保存开启 `stage3_gather_16bit_weights_on_model_save`；所有 rank 都参加保存，不能只在 rank 0 调用模型 gather。

有效 batch：`每卡 batch × gradient_accumulation_steps × WORLD_SIZE`。默认 `1×8×4=32`。增加 GPU 数而不改累积步数会改变有效 batch；公平对比时应固定它或明确报告变化。

## 3. 固定与随机压缩比

编辑 `configs/train.json`：

```json
{
  "compress_ratio": 16,
  "ratio_mode": "random",
  "random_ratios": [4, 8, 16, 32]
}
```

以上是 `model` 节的三个字段，不能替换整个配置文件。

| 字段 | 行为 |
| --- | --- |
| `ratio_mode=fixed` | 每个训练 forward 使用 compress_ratio |
| `ratio_mode=random` | 每个 rank 的每次训练 forward 从数组均匀抽一个比例，同一 microbatch 共用 |
| `random_ratios` | 任意有限数且 >=1；重复元素相当于提高该比例的抽中概率 |
| `compress_ratio` | 固定模式比例，也作为配置中的默认推理参考；benchmark 实际由 --ratios 决定 |

随机抽样使用 torch CPU RNG，随 Trainer 的 native checkpoint 保存/恢复。不同数据/恢复条件不承诺逐位复现；恢复要求同配置、同 backend、同 world size。小于预处理 `min_ratio` 的比例会被拒绝。增减数组不会改动中心选择、聚类和聚合算法。

## 4. 模型参数

| JSON 路径 | 默认 | 含义/修改提示 |
| --- | --- | --- |
| model.name_or_path | Qwen/Qwen3-4B-Instruct-2507 | 两个独立分支的相同初始化基座；Shell 默认覆盖成本地目录 |
| model.revision | main | 远程版本；保存时记录加载到的 commit，本地目录应不可变 |
| model.dtype | bfloat16 | bfloat16 / float16 / float32；需硬件和 DeepSpeed 配置支持 |
| model.attn_implementation | sdpa | eager / sdpa / flash_attention_2；公平对比应统一 |
| model.similarity_chunk_tokens | 16 | 余弦广播按非中心 token 分块；0 恢复原始整块路径 |
| model.lora_r | 128 | encoder 和 decoder 各自的 LoRA rank |
| model.lora_alpha | 32 | LoRA scaling 的 alpha，不能在加载旧 checkpoint 时随意修改 |
| model.lora_dropout | 0.05 | 仅训练生效，推理 eval 关闭 |
| data.prepared_dir | artifacts/prepared/default | 预处理版本目录；Shell PREPARED_DIR 默认覆盖 |

LoRA `bias=none`、`task_type=CAUSAL_LM`、`target_modules=all-linear`。基座参数冻结，保留原始双模型结构，不共享 encoder/decoder 权重。分块余弦仍调用同一个 `CosineSimilarity`；训练时重算分块中间值以避免所有广播中间值同时驻留。数值等价与梯度验证用例见 09；不同 GPU kernel/精度不承诺逐位相同。

## 5. training 参数

| 字段 | 默认 | 作用 |
| --- | --- | --- |
| per_device_train_batch_size | 1 | 每卡 microbatch；先从 1 开始 |
| gradient_accumulation_steps | 8 | 梯度累积次数 |
| learning_rate | 1e-5 | LoRA 优化学习率 |
| num_train_epochs | 1 | max_steps<=0 时按 epoch 训练 |
| max_steps | -1 | 正数时覆盖 epoch，单位 optimizer update |
| lr_scheduler_type | cosine | 学习率曲线，支持 Trainer 提供的合法类型 |
| warmup_ratio | 0.1 | 预热步数比例 |
| weight_decay | 0.01 | 权重衰减 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| gradient_checkpointing | true | encoder/decoder 都使用非重入激活重算 |
| save_strategy / save_steps | steps / 500 | 周期 checkpoint；测试恢复建议先设为 1 |
| save_total_limit | 3 | 周期 checkpoint 保留数量；可能删除较旧 checkpoint |
| logging_steps | 10 | 日志间隔，optimizer steps |
| dataloader_num_workers | 2 | 训练数据加载进程数 |
| dataloader_pin_memory | true | CPU 到 GPU 传输使用 pinned memory |
| seed / data_seed | 42 / 42 | 模型/训练 RNG 与数据 shuffle 种子 |
| report_to | none | 默认仅本地日志，无在线上报 |
| optim | adamw_torch | DDP/普通 ZeRO-2 的优化器；CPU 卸载配置显式定义 DeepSpeed AdamW |
| eval_strategy | no | 不在训练时消费 benchmark；生成测试单独运行 |

其余字段可使用当前 Transformers `TrainingArguments` 支持的选项，但输出目录、deepspeed、bf16/fp16、列保留、save_only_model 等由启动器统一设置，重复配置会报错。此入口没有训练期验证集或 best-model 选择，不要开启 eval_strategy 或 load_best_model_at_end。

## 6. 保存与恢复

```text
artifacts/runs/<RUN_NAME_UTC时间_随机后缀>/
  experiment.json          配置、软件版本、backend、world size、数据指纹
  data_manifest.json       本次训练的 prepared manifest 快照
  checkpoint-N/            双 LoRA 权重、Trainer/native ZeRO 状态、RNG、tokenizer
  final/                   最终双 LoRA 权重与元数据；用于合并/推理
  trainer_state.json       根目录最终 Trainer 状态
  train_results.json      Trainer 保存的训练统计
```

`model.safetensors` 只导出两个分支的 LoRA 张量；推理还需要完全相同的原始基座。普通 DDP checkpoint 保存优化器/scheduler/RNG；ZeRO checkpoint 还包含 DeepSpeed 自己的分片状态，不能只复制小的 safetensors 文件来恢复训练。ZeRO native checkpoint 可能仍很大，应保留整个 checkpoint-N 目录。

```bash
RESUME=artifacts/runs/真实run/checkpoint-500 \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 BACKEND=zero2 \
bash scripts/sh/30_train.sh

# 延长累计目标；新目标必须大于断点 global_step。
RESUME=artifacts/runs/真实run/checkpoint-500 MAX_STEPS=2000 \
bash scripts/sh/30_train.sh
```

Shell 自动以 checkpoint 的父目录作为 RUN_DIR；也可显式指定，但必须一致。恢复检查模型/数据配置、manifest 哈希、backend、world size，只有累计 max_steps 可变。使用远程 main 的恢复会继续使用断点中保存的 commit。改变模型、LoRA、压缩比数组、候选池或显卡进程数时开新实验。`final/` 和旧版导入目录没有 native optimizer 状态，不能用于 `RESUME`。

多节点需要所有节点相同且共享可见的 `RUN_DIR`、模型和数据路径；设置相同的 `NNODES/NPROC_PER_NODE/MASTER_ADDR/MASTER_PORT`，各节点使用不同 `NODE_RANK`。该入口使用 torchrun 静态 rendezvous，通信网络与 NCCL 环境由集群配置。先完成单节点验证再扩展。
