# 03 ICAE 任务微调、合并与评测

在 Linux 项目根目录执行。先准备所选任务的监督数据，并记录预处理使用的模型快照及输出目录。[Benchmark](../benchmark/03_preprocessing.md) · [NIAH](../niah/03_prepare_train.md) · [RULER QA](../ruler_qa/03_preprocessing.md)。

```bash
task=niah  # benchmark / niah / ruler_qa
train="scripts/$task/sh/icae/30_train.sh"
model=/path/to/base_model_snapshot
prepared=/path/to/task_prepared
```

## 从 ICAE 预训练适配器开始

按 [02 预训练](02_pretraining.md) 生成适配器，使用其 `final/` 或 `checkpoint-N/`。以下命令只做预检。确认输出中的 `initialization_mode` 为 `pretrained`、`initialization.checkpoint` 指向预期目录后，移除 `PLAN_ONLY=1` 启动训练。

```bash
INIT_FROM=/path/to/pretrain_run/final MODEL_PATH="$model" PREPARED_DIR="$prepared" \
    CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp PLAN_ONLY=1 bash "$train"
```

微调载入预训练的 encoder LoRA、记忆槽、LM 和 AE 标记，并创建微调优化器。LoRA、记忆槽和 LM 标记参与任务训练，AE 标记随权重保存并冻结。预检比较基座路径及 revision、tokenizer 指纹、LoRA rank、alpha 和目标模块、槽数及末块策略，并记录来源权重和实验文件的 SHA256。两个阶段使用同一本地快照路径，或同一 Hub ID 与 commit。

## 从原始基座直接微调

设置 `INIT_FROM_BASE=1`，并让 `MODEL_PATH` 指向分词时使用的原始模型快照。训练载入基座权重，按配置新建 encoder LoRA、记忆槽和 LM 标记，使用任务答案作为监督。默认配置还保留一个新初始化、冻结的 AE 标记。预检输出的 `initialization_mode` 应为 `base`。移除 `PLAN_ONLY=1` 启动训练。

```bash
INIT_FROM_BASE=1 MODEL_PATH="$model" PREPARED_DIR="$prepared" \
    CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp PLAN_ONLY=1 bash "$train"
```

## 恢复微调

```bash
RESUME=/path/to/sft_run/checkpoint-500 MAX_STEPS=1200 \
    MODEL_PATH="$model" PREPARED_DIR="$prepared" \
    CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp bash "$train"
```

`RESUME`、`INIT_FROM` 和 `INIT_FROM_BASE=1` 互斥。续训恢复当前微调 run 的 Trainer 和优化器状态，并沿用已记录的初始化来源。保持配置、数据、后端和总进程数一致。使用 `MAX_STEPS` 延长训练时，填入大于已完成步数的累计目标。若曾导出初始化变量，续训前执行 `unset INIT_FROM INIT_FROM_BASE`。

## 合并与评测

```bash
sft=/path/to/sft_run/final
CHECKPOINT="$sft" bash "scripts/$task/sh/icae/40_merge_lora.sh"
CHECKPOINT="${sft}_merged" RATIOS='32' bash "scripts/$task/sh/icae/50_benchmark.sh"
```

评测也可用 `CHECKPOINT="$sft"` 在内存中合并。任务测试数据、热力图及 Judge 命令见 [评测指南](../04_evaluation.md) 和 [LLM-as-Judge](../05_llm_judge.md)。默认微调配方固定 32×，评测比例可独立设置。改变训练比例时核对预处理的 `min_ratio` 和解码预算。两阶段复现使用与预训练一致的比例。
