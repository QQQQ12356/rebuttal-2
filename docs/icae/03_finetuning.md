# 03 ICAE 任务微调、合并与评测

先完成预训练，然后选择任务并准备其监督数据。[Benchmark 数据](../benchmark/03_preprocessing.md) · [NIAH 数据](../niah/03_prepare_train.md) · [RULER QA 数据](../ruler_qa/03_preprocessing.md)。

```bash
task=niah                                      # benchmark / niah / ruler_qa
pretrained=/path/to/pretrain_run/final
train="scripts/$task/sh/icae/30_train.sh"

INIT_FROM="$pretrained" CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp \
    PLAN_ONLY=1 bash "$train"

INIT_FROM="$pretrained" CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp \
    bash "$train"
```

`INIT_FROM` 或 `--init-from` 传入预训练 `final/` 或 `checkpoint-N/`。微调加载 encoder LoRA、记忆槽和两个标记，并为新 run 创建优化器。AE 标记保存在微调权重中并冻结，LM 标记用于答案训练。预检比较基座路径、revision、tokenizer 指纹、LoRA rank/alpha/目标模块、槽数和末块策略。实验元数据记录预训练权重与实验文件的 SHA256。预训练与微调使用同一个本地模型快照路径，或者同一个 Hub ID 与 commit。

恢复已启动的微调时改用 `RESUME=/path/to/sft_run/checkpoint-N`。恢复使用微调 run 自己的状态和已记录的预训练来源。默认微调配方使用固定 32×，评测可单独指定 `RATIOS`。需要改变训练压缩比时，同时核对任务预处理的 `min_ratio` 和解码预算。正式复现使用和预训练一致的比例。

```bash
sft=/path/to/sft_run/final
CHECKPOINT="$sft" bash "scripts/$task/sh/icae/40_merge_lora.sh"
CHECKPOINT="${sft}_merged" RATIOS='32' bash "scripts/$task/sh/icae/50_benchmark.sh"
```

评测也可直接读取 `final/` 并在内存中合并。NIAH 和 RULER QA 的测试数据、热力图及 Judge 命令见 [评测指南](../04_evaluation.md) 和 [LLM-as-Judge](../05_llm_judge.md)。现有 ICAE 微调检查点仍依据其自带模型配置加载，新建微调 run 默认要求预训练权重。
