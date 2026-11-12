# 02 ICAE 预训练

先完成 [环境和数据准备](01_data.md)。以下命令默认读取 `prepared/v1`。若预处理使用其他目录，预训练时传入同一 `PREPARED_DIR`。

```bash
PLAN_ONLY=1 bash scripts/icae/sh/30_pretrain.sh

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKEND=ddp MAX_STEPS=1 \
    bash scripts/icae/sh/30_pretrain.sh

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BACKEND=ddp \
    bash scripts/icae/sh/30_pretrain.sh
```

`PLAN_ONLY=1` 验证配置、数据清单及文件哈希，输出有效 batch 和 token 长度预算。模型实际位置上限在训练初始化时校验。单步检查会产生独立 run。正式运行日志打印其路径，完成后将该 run 的 `final/` 传给微调的 `INIT_FROM`。

每步从前半段编码一次记忆。Decoder 通过 AE 标记预测重建目标，通过 LM 标记预测续写目标。目标均以 teacher forcing 计算 token 交叉熵，批内 padding 不计入损失。联合损失为 `(ae_weight × AE + lm_weight × LM) / (ae_weight + lm_weight)`。默认权重各为 1。把某项设为 0 可关闭该项。使用默认两项预训练时，交接给微调的 LM 标记已经过续写监督。

| 文件 | 作用 |
| --- | --- |
| `experiment.json` | 预训练配置、数据清单哈希与运行环境 |
| `checkpoint-N/` | 周期保存，含用于恢复的 Trainer/优化器状态 |
| `final/` | 适配器、记忆槽、两个标记、tokenizer 和实验元数据 |
| `eval_results.json` | 有验证集时，在训练结束后记录联合验证损失 |

```bash
RESUME=/path/to/pretrain_run/checkpoint-500 MAX_STEPS=1200 \
    CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BACKEND=ddp \
    bash scripts/icae/sh/30_pretrain.sh
```

恢复同一 run 时保持基座、数据、配置、后端和进程数一致，`MAX_STEPS` 是累计更新目标。多卡有效 batch 为每卡 batch × 梯度累积步数 × 进程数。支持单机 DDP、ZeRO-2 和 ZeRO-2 CPU 卸载。若显存不足，优先从每卡 batch 1 开始，必要时缩小文档长度上限并重新预处理。
