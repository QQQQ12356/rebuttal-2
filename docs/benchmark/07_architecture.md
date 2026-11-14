# 07 架构、加速来源与编码规范

## 1. 原始方法保持

当前基线是用户放入本目录的 `modeling_seco_cluster.py`，原文件 SHA256：

```text
37e7ed3ed8746b61bcaffcb3a81994ec45d70872aae08bf1e0b0408374a61d6f
```

原文件保存在 `legacy/py/modeling_seco_cluster.py`。`seco/compression.py` 沿用其中的中心选择、组分配和加权聚合逻辑，并支持可选的分组统计。静态检查去除统计代码后比较两个压缩方法的 AST。`tests/reference_compression.py` 保存冻结基线，供 Linux 数值与梯度验证。

原始压缩过程：

1. 从 encoder 的最后隐藏状态分离有效 context 与 prompt token；padding 不参与任何计算。
2. prompt token 均值构造问题表示；无 prompt 时用 context 均值。
3. 令 `K=min(L,max(2,ceil(L/R)))`，按 context 到 prompt 的余弦相似度全局 Top-K 选中心，保留原 Top-K 顺序。
4. 非中心 token 按到各中心的余弦相似度 argmax 分配；不根据乘积 score 改变分配。
5. 中心自身权重 score 为到 prompt 的相似度；成员 score 为到中心相似度乘到 prompt 相似度；在各簇内做 softmax，再加权求和。
6. 按原中心顺序拼接压缩向量，后接完整 prompt 隐藏状态，送入独立 decoder。

训练仍为原始 teacher forcing 与忽略压缩前缀标签的 causal LM loss。context/prompt 仍分别分词后拼接，不引入新的聊天模板或改变有效 prompt 范围；没有将 SeCo 改为离散 token 丢弃、文本摘要、随机中心、分块选中心或向量检索。

## 2. 来自 test_fast 的工程优化

| 来源/做法 | 新实现 | 语义约束 |
| --- | --- | --- |
| test_fast 预先合并 LoRA | PEFT 官方 merge_and_unload；HF encoder/decoder 独立导出 | 使用配置中的 alpha/r；不手工改 key；两分支独立 |
| test_fast encoder.model + last_hidden_state | 只调用带 LoRA 层的 backbone | 不计算 encoder 词表 logits，不返回所有层，不建立 encoder KV cache |
| test_fast inference_mode | generate_batch 上统一开启 | 不关闭训练梯度 |
| test_fast 多卡分片 | 每 rank 不重复样本，写 shard 后校验合并 | 不因补齐 sampler 重复算分；支持空 shard |
| test_fast worker/pinned memory/预热 | DataLoader、non_blocking H2D、可控 warmup | 不跳过前若干真实测试样本 |
| 原始可选 attention 加速意图 | 显式传 attn_implementation，默认 sdpa | FlashAttention 必须真实安装可用，不保留无效果开关 |

没有照搬 `test_fast` 只取 `hidden_states[0]` 的 batch=1 写法、context+prompt 重新整体分词的边界错误、固定 10-token 生成上限、静默 strict=False 忽略真正缺权重，或只声明不执行的 torch.compile 选项。当前未提供 torch.compile，以免对动态聚类/Python 循环承诺未验证的加速。

另外修正了原始 padding 的两点工程问题：attention_mask 按长度构造，真实 EOS 不因 PAD 共用 ID 被误屏蔽；teacher forcing 和生成之前将压缩前缀左 padding，使第一个答案总是紧跟最后一个有效压缩 token。训练使用显式连续 position_ids。该修正保持有效 token 的压缩结果，修复不等长 batch 的标签对齐。

长 context 的原始余弦广播会产生很大的 N×K×D 中间张量。`seco/similarity.py` 按非中心 token 切块，仍逐元素调用相同 CosineSimilarity 并原顺序拼回完整 N×K 相似度矩阵；不做近似、不重排中心、不改 argmax。训练重算该分块临时计算。`similarity_chunk_tokens=0` 可切回原始路径；不同 dtype/kernel 的浮点舍入或相似度平局仍应在真实 GPU 上验证。

## 3. 模块职责

| 文件 | 职责 |
| --- | --- |
| seco/compression.py | 两段语义压缩和可选的逐样本分组计数 |
| seco/similarity.py | 等公式的余弦计算内存控制 |
| seco/model.py | 双模型、LoRA、前缀 padding、训练 forward、生成、权重加载/合并 |
| seco/data.py | 各源适配、分词边界、采样/长度规则、collator |
| seco/training.py | Trainer 双 adapter 保存与 native checkpoint 恢复衔接 |
| seco/metrics.py | 统一 EM/F1、micro/macro、重复 ID 检查 |
| seco/reporting.py | 结果协议校验、Excel 汇总与长文本导出 |
| seco/downloads.py | 固定 split/config 分片选择、网络重试、源分片缓存与本地记录读取；不参与模型压缩 |
| seco/io.py / config.py | 原子文本写入、哈希、路径与配置约束 |
| scripts/benchmark/py | 命令行流程编排；不复制核心算法 |
| scripts/benchmark/sh | Linux 环境变量、可编辑启动设置和日志 |

## 4. 本次采用的编码 skill

按照用户指定的 [mattpocock/skills](https://github.com/mattpocock/skills)，实际采用 [codebase-design/SKILL.md](https://github.com/mattpocock/skills/blob/main/skills/engineering/codebase-design/SKILL.md) 的模块设计规范：用小接口集中真实复杂逻辑，数据适配、模型生命周期、指标与导出各有明确维护位置；调用方和测试通过相同公开接口访问。统一领域名称见本页和各参数表，不引入只转发调用的配置层。

未全局安装该仓库，也未启动其 issue tracker、PR 发布、强制访谈或其他与本次本地代码改造无关的工作流。Hugging Face 下载约定参考当前可用的 `hf-cli` skill；文档使用新 `hf` 命令，程序使用对应 Hub SDK 以记录 commit。测试原则是验证用户可观察行为与冻结原始结果；只编写与本任务相关的验证，不声明测试已经在 Windows/GPU 上运行。
