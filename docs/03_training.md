# 03 训练、恢复与合并

以下命令以 COMI 的 NIAH 单卡验证为例。切换任务时把 `task` 改为 `benchmark` 或 `ruler_qa`，切换方法时改 `method`。准备好该任务的训练数据后执行。

ICAE 新建微调 run 时可用 `INIT_FROM` 载入预训练适配器，也可用 `INIT_FROM_BASE=1` 从原始基座开始。[ICAE 训练指南](icae/00_README.md)。

```bash
method=comi
task=niah
step="scripts/$task/sh/$method/30_train.sh"

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp PLAN_ONLY=1 bash "$step"
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp MAX_STEPS=1 bash "$step"
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp bash "$step"
```

`PLAN_ONLY=1` 检查任务、文件哈希、比例与配置并输出计划。第一步真实训练用于核对损失和权重保存，正式训练使用新的运行目录。运行日志打印实际路径。四种方法均支持 DDP、ZeRO-2 和 ZeRO-2 CPU 卸载。SeCo 还支持 ZeRO-3 CPU 卸载。三种新增方法每个样本可能执行不同次数的分段编码，ZeRO-3 参数收集要求各 rank 以一致顺序进入子模块。

| 调整目标 | 修改位置 |
| --- | --- |
| 固定或随机比例 | `configs/<任务>/<方法>/train.json` 中的 `model.ratio_mode`、`compress_ratio`、`random_ratios` |
| 学习率、累积步数和 batch | 同一文件的 `training` |
| 单卡／多卡 | `CUDA_VISIBLE_DEVICES` 与 `NPROC_PER_NODE`，有效 batch 为每卡 batch × 累积步数 × rank 数 |
| 多机 | 每节点独立设置 `NODE_RANK`，共用 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`、`RUN_DIR` 与数据路径 |
| 恢复 | `RESUME=<run>/checkpoint-N`，保持原方法、数据、配置和并行设置。可通过 `MAX_STEPS` 提高累计训练目标 |

## 并行运行的默认端口

| 训练任务 | SeCo | COMI | ICAE | RAM |
| --- | --- | --- | --- | --- |
| Benchmark | 29500 | 29501 | 29502 | 29503 |
| NIAH | 29600 | 29601 | 29602 | 29603 |
| RULER QA | 29610 | 29611 | 29612 | 29613 |

ICAE 预训练使用 29620。重复并行启动同一个任务和方法时，为另一份 run 设置独立端口，例如 `MASTER_PORT=29701 bash scripts/niah/sh/comi/30_train.sh`。同一个多机 run 的所有节点使用相同端口。并行 run 的 GPU 可通过 `CUDA_VISIBLE_DEVICES` 分配。评测脚本使用 torchrun 的 `--standalone` 自动选择端口。

`checkpoint-N` 用于原生恢复。`final` 保存推理用 LoRA 和方法参数，需要原基座。运行 `40_merge_lora.sh` 可将其导出为本地自包含模型，融合分支与记忆/对齐参数随模型一起保存。

```bash
CHECKPOINT=/path/to/run/final MERGE_DEVICE=cpu bash "scripts/$task/sh/$method/40_merge_lora.sh"
# 默认写到 /path/to/run/final_merged
```

方法专有参数和调节建议见 [06 参数](06_parameters.md)。优化器、保存和恢复细节见各任务训练指南。[Benchmark](benchmark/04_training.md) · [NIAH](niah/04_training.md) · [RULER QA](ruler_qa/04_training.md)。
