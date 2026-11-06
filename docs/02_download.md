# 02 模型与数据下载

## 1. 默认模型

默认仓库为 [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，官方说明该版本仅支持非思考模式。代码使用完整仓库名，避免把不存在的简称当作 Hub ID。

```bash
bash scripts/sh/01_download_model.sh
# 指定 commit 和保存位置：
REVISION=你的commit MODEL_PATH=/models/qwen3_4b bash scripts/sh/01_download_model.sh
```

`01_download_model.py` 将 `main` 解析为 commit SHA 后下载模型、tokenizer 与配置，保存 `download.json`。切换模型时同时修改预处理与训练的 `MODEL_PATH`，重新生成 prepared 数据。训练保存远程模型实际加载到的 commit；本地模型应保留下载元数据并保持目录不可变。

## 2. 来源注册表

编辑 `configs/datasets.json` 更换来源、config、split 或 revision。NaturalQA 在本项目中指 Natural Questions，标准名称统一写作 `naturalqa`。

| 角色 | 项目名称 | 默认 repo | config / split | 转换说明 |
| --- | --- | --- | --- | --- |
| 训练 | NaturalQA | `google-research-datasets/natural_questions` | default / train | 从全文 token 中去掉 HTML；仅保留 short-answer/yes-no 可回答样本 |
| 训练 | 2WikiMQA | `framolfese/2WikiMultihopQA` | default / train | 保持所有 context；HF 再发布版本，注册表可换为同 schema 来源 |
| 训练 | HotpotQA | `hotpotqa/hotpot_qa` | distractor / train | 保留干扰段落，禁止依据 supporting facts 筛选输入 |
| 训练 | NarrativeQA | `deepmind/narrativeqa` | default / train | 使用 `document.text` 完整故事，超长过滤，不替换为人工摘要 |
| 训练 | MultiNews | `tau/multi_news` | default / train | 使用 HF 的 `refs/convert/parquet` 转换分支，避免执行旧 dataset script |
| 测试 | NaturalQA | `Twwilght/NQ` | default / test | 读取 question / ctxs / answers |
| 测试 | 2WikiMQA | `Twwilght/MQA` | default / test | 同上 |
| 测试 | HotpotQA | `Twwilght/HQA` | default / test | 同上 |
| 测试 | MultiNews | `Twwilght/NEWS` | default / test | 保留源 question 中的摘要指令与所有新闻 |

训练来源不是从上述四个 Twwilght 测试集切分出来的。NarrativeQA 测试集不下载、不计分。来源资料：[Natural Questions schema](https://huggingface.co/datasets/google-research-datasets/natural_questions)、[2Wiki 再发布说明](https://huggingface.co/datasets/framolfese/2WikiMultihopQA)、[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)、[NarrativeQA](https://huggingface.co/datasets/deepmind/narrativeqa)、[MultiNews 源仓库](https://huggingface.co/datasets/tau/multi_news)。

## 3. 下载命令与容量控制

```bash
bash scripts/sh/10_download_data.sh
ROLE=benchmark bash scripts/sh/10_download_data.sh
ROLE=training TRAIN_LIMIT=10000 bash scripts/sh/10_download_data.sh

# 单独下载一个训练源。
ROLE=training bash scripts/sh/10_download_data.sh --datasets narrativeqa
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--registry` | configs/datasets.json | 注册表路径，相对路径基于项目根 |
| `--output` | artifacts/raw | 下载输出根目录 |
| `--role` | all | all / training / benchmark |
| `--datasets` | 当前 role 的全部源 | canonical 名称数组；all 模式下须在两个 role 中都存在 |
| `--train-limit` | 30000 | 每个训练源读取前 N 条原始行；0 读取完整 split |

流式读取减少本地 Arrow 缓存，但底层 Parquet 分片仍可能整片下载。Natural Questions 原始规模很大，NarrativeQA 全文也可能占用大量空间。`TRAIN_LIMIT` 是原始源前缀限制，**不是从完整数据集均匀采样**；后续预处理只在这批候选中做均匀 reservoir sampling。做最终实验时应明确报告候选池范围，或设 `TRAIN_LIMIT=0` 扫描完整训练 split。

下载每个源生成：

```text
artifacts/raw/training/naturalqa.jsonl
artifacts/raw/training/naturalqa.source.json
artifacts/raw/benchmark/naturalqa.jsonl
artifacts/raw/benchmark/naturalqa.source.json
```

`.source.json` 记录 repo/config/split、解析后的 commit、行数与文件 SHA256。相同身份且哈希匹配会复用；身份变化或不完整文件会拒绝覆盖，请换 `RAW_DIR`。不要手动修改现有 raw 文件后继续使用原 manifest。

## 4. 镜像与本地原始文件

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/sh/10_download_data.sh
```

若 MultiNews 转换分支在当前 endpoint 不可用，可以切回官方 endpoint；也可在 Linux 自行从合法来源准备 JSONL，然后放入 `RAW_DIR/training/multinews.jsonl`，每行包含 `document` 和 `summary`。其他源同理，文件名与 schema 必须符合注册名称。没有 `.source.json` 的本地输入会明确记为 `local_file` 加实际 SHA256，不伪造 Hub 来源。

已有 SeCo 标准数据可直接使用 `input` / `prompt` / `answer` 格式放入对应 raw 文件；answer 可以是字符串或非空字符串数组。不要将测试样本放入 training 目录。自定义适配器只需修改 `seco/data.py::normalize_record`，保留统一输出协议。
