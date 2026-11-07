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

数据写入 `artifacts/ruler_qa/raw/`。脚本逐文件按 `configs/ruler_qa/sources.json` 的顺序尝试原站、Hugging Face 和 HF-Mirror。遇到连接失败、下载失败或 SHA-256 不匹配时自动尝试下一个地址，通过校验后才写入目标文件。已存在且校验成功的文件会复用。[HF-Mirror 使用说明](https://hf-mirror.com/)

| 文件 | 镜像来源与校验 |
| --- | --- |
| SQuAD 2.0 train、dev | `GEM/squad_v2` 的两份原始 JSON，固定 revision，SHA-256 均与原站相同。[SQuAD 镜像](https://huggingface.co/datasets/GEM/squad_v2/tree/eac7d401bef8f37ea4037c10f6408e84853c667b/squad_data) |
| HotpotQA train | `namlh2004/hotpotqa` 的 `hotpot_train_v1.json` 与官方 v1.1 的 SHA-256 相同，落盘仍命名为 `hotpot_train_v1.1.json`。[HotpotQA 训练镜像](https://huggingface.co/datasets/namlh2004/hotpotqa/blob/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_train_v1.json) |
| HotpotQA dev distractor | 优先使用与官方 SHA-256 一致的 `RAGLAB/data` 镜像，再尝试具有独立 SHA-256 的 `namlh2004/hotpotqa` 镜像。预处理清单记录实际使用的文件摘要。[RAGLAB dev 镜像](https://huggingface.co/datasets/RAGLAB/data/blob/c33b09b7fa5099b89830e84f9bdff78329268d9b/eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json) · [备用 dev 镜像](https://huggingface.co/datasets/namlh2004/hotpotqa/blob/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json) |

- 自定义磁盘路径时设置 `RAW_DIR=/data/ruler/raw`，下载与预处理使用同一个值。
- 如需优先使用 HF-Mirror 或内网镜像，复制源配置，把对应地址移到每个文件列表的首位，再用 `SOURCES=/path/to/sources.json bash scripts/ruler_qa/sh/10_download_data.sh` 执行。保持四个目标文件名和各地址配套的 SHA-256。
- 文件校验失败时，把损坏文件移到其他路径再重跑。离线复制文件时，先用 `sha256sum` 与配置中的摘要比对。
