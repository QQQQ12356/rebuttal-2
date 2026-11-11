# 04 训练与合并

```bash
export RUN_DIR=artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/runs/qa_main_v1
PLAN_ONLY=1 bash scripts/ruler_qa/sh/seco/30_train.sh
bash scripts/ruler_qa/sh/seco/30_train.sh
```

默认单 GPU DDP，batch 1、梯度累积 8、训练 1 epoch。输出 `final/` 是编码器和解码器的 LoRA 权重、tokenizer 和实验元数据。`checkpoint-N/` 还包含原生训练状态，供续训。标准日志位于 `artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/logs/`，计划模式检查配置、预处理清单与训练数据哈希。

| 模式 | 配置方法 | 实际行为 |
| --- | --- | --- |
| 随机压缩（默认） | `ratio_mode: "random"`，`random_ratios: [4, 8, 16, 32]` | 每次训练 forward 从数组均匀抽一个值，同一 forward 内整批使用该值。梯度累积中的各次 forward 可使用不同值 |
| 固定压缩 | `ratio_mode: "fixed"`，修改 `compress_ratio` | 全程使用单个压缩比 |
| 显存调整 | 减少训练长度、保持 batch 1、开启梯度检查点或选 ZeRO | 有效 batch 为 `NNODES × NPROC_PER_NODE × per_device_train_batch_size × gradient_accumulation_steps` |

SeCo 的压缩比越大，目标保留中心越少。训练数组的最小值需 ≥ 训练预处理的 `min_ratio`。使用固定和随机两组实验时，分别保存配置副本并使用不同 `RUN_DIR`，保持其余条件一致。

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BACKEND=zero2   RUN_DIR=artifacts/seco/Llama-3.2-1B-Instruct/ruler_qa/runs/qa_zero2_v1 bash scripts/ruler_qa/sh/seco/30_train.sh

RESUME="$RUN_DIR/checkpoint-50" MAX_STEPS=200 bash scripts/ruler_qa/sh/seco/30_train.sh
```

上面的续训路径需要替换成实际存在的检查点。续训保持模型、数据、训练配置、后端和 world size 一致。`MAX_STEPS` 表示累计 optimizer step 上限。DDP、ZeRO-2、ZeRO-2 CPU offload、ZeRO-3 CPU offload 分别使用 `ddp`、`zero2`、`zero2_cpu`、`zero3_cpu`。ZeRO 需要安装额外依赖。

可以直接用适配器运行评测，评测进程默认在内存中合并。若要重复评测或迁移模型，可先导出合并后的编码器和解码器。

```bash
CHECKPOINT="$RUN_DIR/final" bash scripts/ruler_qa/sh/seco/40_merge_lora.sh
```

默认 CPU 合并，输出 `$RUN_DIR/final_merged`。可用 `MERGE_DEVICE=cuda`，或用 `MERGED_DIR` 指定新路径。基座权重迁移后，用 `MODEL_PATH` 指向同一版本的本地权重。
