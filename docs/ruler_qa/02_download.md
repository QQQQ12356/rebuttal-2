# 02 模型与数据下载

```bash
bash scripts/ruler_qa/sh/01_download_model.sh
bash scripts/ruler_qa/sh/10_download_data.sh
```

模型默认写入 `artifacts/models/Qwen3-4B-Instruct-2507/`。下载脚本先解析 Hub revision，再记录真实 commit 到 `download.json`。复现实验可显式设置 `REVISION=<commit>` 并将不同版本存到不同 `MODEL_PATH`。已有同版本本地快照时，可直接把 `MODEL_PATH` 指向它。替换模型需重新预处理并重新训练。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

| 任务 | 训练来源 | 评测来源 |
| --- | --- | --- |
| `qa_1` | SQuAD 2.0 `train-v2.0.json`，仅可回答样本 | SQuAD 2.0 `dev-v2.0.json`，仅可回答样本 |
| `qa_2` | HotpotQA `hotpot_train_v1.1.json` | HotpotQA `hotpot_dev_distractor_v1.json` |

测试采用公开 dev 上有参考答案的题。训练长文档数据由原始训练 split 在本地构造。下载同时保留 dev，便于训练预处理核对重复问题。SQuAD 和 HotpotQA 均由源项目以 CC BY-SA 4.0 发布。[SQuAD 数据](https://rajpurkar.github.io/SQuAD-explorer/) · [HotpotQA 数据](https://hotpotqa.github.io/) · [RULER 的 QA 下载脚本](https://github.com/NVIDIA/RULER/blob/main/scripts/data/synthetic/json/download_qa_dataset.sh)

数据写入 `artifacts/ruler_qa/raw/`。脚本按 `configs/ruler_qa/sources.json` 顺序尝试 URL，校验 SHA-256 后发布。已存在且校验成功的文件会复用。HotpotQA 使用固定 revision 的镜像作后备，镜像训练文件的上游名称为 `hotpot_train_v1.json`，SHA-256 与原始 v1.1 相同，脚本保存为标准本地文件名。dev 镜像有独立的 SHA-256，预处理清单记录实际使用的文件摘要。[HotpotQA 训练镜像](https://huggingface.co/datasets/namlh2004/hotpotqa/blob/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_train_v1.json) · [HotpotQA dev 镜像](https://huggingface.co/datasets/namlh2004/hotpotqa/blob/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json)

- 自定义磁盘路径时设置 `RAW_DIR=/data/ruler/raw`，下载与预处理使用同一个值。
- 内网镜像用 `SOURCES` 指向一份 JSON 副本，保持四个目标文件名。每个条目是按优先级排列的 `[URL, SHA-256]` 列表。
- 文件校验失败时，把损坏文件移到其他路径再重跑。离线复制文件时，先用 `sha256sum` 与配置中的摘要比对。
