# 00 方法、代码与目录

| 方法 | 压缩和解码 | 本项目训练配置 |
| --- | --- | --- |
| SeCo | 上下文整体编码，选语义中心并进行聚类加权，随后拼接 prompt | Encoder 和 decoder 各自训练 LoRA，原有压缩函数沿用 |
| COMI | 每段拼接 query 后编码。粗粒度依据相关性减冗余分配组宽，细粒度依据同类分数做组内加权聚合，随后经 LSA 融合 | 默认 8 层 encoder、1 层 LSA。Encoder、decoder、LSA 训练 LoRA。组宽取整时每组至少一个 token，以固定槽数对齐实验 |
| ICAE | 每个 chunk 追加可学习记忆 token，取对应隐藏向量。默认尾块使用完整槽数 | 先做 AE 重建和 LM 续写预训练，再载入权重做任务微调。Encoder 使用 Q/V LoRA，decoder 冻结 |
| RAM | Query 与短段分别编码。高相关段保留原始 token，其余段按 query 相关性池化，经融合及线性对齐后恢复段序 | 默认保留 5% 的段、8 层 encoder、1 层融合。训练 LoRA、线性对齐及基于答案 token 匹配的对比损失 |

COMI 和 RAM 采用本项目的 LoRA 配方。RAM 的对比损失使用训练标签中主答案的精确 token 匹配，命中跨段时会标记涉及的所有段。对比论文或上游指标时应记录训练配方、模型版本、保留比例与压缩预算。[COMI 参考实现](https://github.com/Twilightaaa/COMI/tree/d175a44eaf71084cb1ba2edc91e3f18944e971ad) · [RAM 参考实现](https://github.com/Twilightaaa/RAM/tree/1e82e3bf207222e2d5c9165335bbb1f4e471ca20) · [ICAE 两阶段实验](icae/00_README.md)。

```text
seco/ | comi/ | icae/ | ram/             方法实现
scripts/<任务>/sh/<方法>/                 每个步骤的 Bash 入口和可修改默认值
scripts/<任务>/py/<方法>/                 每个步骤的 Python 入口和 DEFAULT_ARGS
configs/<任务>/<方法>/train.json          方法参数、LoRA 和优化器
artifacts/<方法>/<基座>/model/            模型快照
artifacts/<方法>/<基座>/<任务>/raw/       原始数据
artifacts/<方法>/<基座>/<任务>/prepared/  分词结果、数据清单和校验信息
artifacts/<方法>/<基座>/<任务>/runs/      训练配置、检查点和合并模型
artifacts/<方法>/<基座>/<任务>/benchmarks/ 预测、指标和 Judge 结果
artifacts/<方法>/<基座>/<任务>/logs/      日志
artifacts/<方法>/<基座>/benchmark/reports/ Excel
artifacts/hf_cache/                     Hugging Face 共用缓存
```

任务名取 `benchmark`、`niah` 或 `ruler_qa`，方法名取 `seco`、`comi`、`icae` 或 `ram`。编号入口调用任务目录内的共享实现。旧数据可通过 `RAW_DIR`、`PREPARED_DIR`、`GRID_DIR` 或 `TEST_DIR` 引用。SeCo 旧检查点缺少 `model.method` 时按 SeCo 加载，使用与检查点匹配的基座和 tokenizer。
