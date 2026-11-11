# 00 ICAE 两阶段实验

ICAE 默认使用 `meta-llama/Llama-3.2-1B-Instruct`。先用无标注文档预训练一次，再分别为 Benchmark、NIAH 和 RULER QA 建立微调 run。下载、预处理、训练和评测均在 Linux 上执行。

| 编号 | 指南 | 内容 |
| --- | --- | --- |
| 01 | [环境与预训练数据](01_data.md) | 安装、模型快照、语料格式、AE/LM 目标 |
| 02 | [预训练](02_pretraining.md) | 固定压缩比、预检、训练、验证与恢复 |
| 03 | [任务微调与评测](03_finetuning.md) | 权重交接、任务数据、合并、测试与 Judge |
| 04 | [参数索引](04_parameters.md) | Shell、Python CLI、模型和优化器配置 |

| 阶段 | 训练信号 | 训练参数 |
| --- | --- | --- |
| 预训练 | AE 重建文档前半段，LM 预测文档后半段。两个目标共用同一份压缩记忆 | Encoder 的 Q/V LoRA、记忆槽、AE 和 LM 标记 |
| 微调 | 压缩任务上下文，拼接 LM 标记和问题，预测任务答案 | 从预训练恢复 Encoder LoRA、记忆槽及两个标记。继续训练 LoRA、记忆槽和 LM 标记 |

两个阶段均冻结 decoder。默认每块使用 32 个记忆槽，固定压缩比为 32，末块也使用完整槽数。任务输入沿用项目的 prompt 和答案分词格式，位置编码沿用项目的连续位置。方法参考同级 `../ICAE` 的 `pretrain/pre_prepare_data.py`、`model/modeling.py` 和 `sft/instruction_trainer.py`。新训练使用 PEFT 与 Safetensors 保存，交接使用本项目预训练入口产出的适配器。

```text
scripts/icae/{py,sh}/                             ICAE 预训练入口
configs/icae/pretrain.json                       预训练参数
configs/<任务>/icae/train.json                    微调参数
artifacts/icae/<基座>/model/                      共用模型快照
artifacts/icae/<基座>/icae_pretrain/{raw,prepared,runs,logs}/
artifacts/icae/<基座>/<任务>/{raw,prepared,runs,benchmarks,logs}/
```
