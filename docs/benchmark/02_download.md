# 02 模型与数据下载

## 1. 默认模型

默认仓库为 [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)。代码使用完整仓库名，避免把不存在的简称当作 Hub ID。

```bash
bash scripts/benchmark/sh/seco/01_download_model.sh
# 指定 commit 和保存位置：
REVISION=你的commit MODEL_PATH=/models/llama32_1b bash scripts/benchmark/sh/seco/01_download_model.sh
```

`01_download_model.py` 将 `main` 解析为 commit SHA 后下载模型、tokenizer 与配置，保存 `download.json`。切换模型时同时修改预处理与训练的 `MODEL_PATH`，重新生成 prepared 数据。训练保存远程模型实际加载到的 commit；本地模型应保留下载元数据并保持目录不可变。

## 2. 来源注册表

编辑 `configs/benchmark/datasets.json` 更换来源、config、split 或 revision。NaturalQA 在本项目中指 Natural Questions，标准名称统一写作 `naturalqa`。

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
bash scripts/benchmark/sh/seco/10_download_data.sh
ROLE=benchmark bash scripts/benchmark/sh/seco/10_download_data.sh
ROLE=training TRAIN_LIMIT=10000 bash scripts/benchmark/sh/seco/10_download_data.sh

# 单独下载一个训练源。
ROLE=training bash scripts/benchmark/sh/seco/10_download_data.sh --datasets narrativeqa
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--registry` | configs/benchmark/datasets.json | 注册表路径，相对路径基于项目根 |
| `--output` | artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw | 下载输出根目录 |
| `--role` | all | all / training / benchmark |
| `--datasets` | 当前 role 的全部源 | canonical 名称数组；all 模式下须在两个 role 中都存在 |
| `--train-limit` | 30000 | 每个训练源读取前 N 条原始行；0 读取完整 split |
| `--download-mode` | local | 先完整下载分片，再本地解码；stream 保留旧远端流式路径 |
| `--source-cache` | artifacts/seco/Llama-3.2-1B-Instruct/benchmark/source_cache | 分片缓存根目录，保存完整文件及可续传的 .incomplete |
| `--download-attempts` / `--retry-delay` | 6 / 2 | 每文件/元数据调用的外层尝试次数>=1；初始退避0–30秒，后续退避最大30秒；SDK内部仍可能重试 |
| `--metadata-timeout` | 60 | dataset-info API 超时，单位秒，范围0<值<=3600；独立于文件下载/ETag超时 |
| `--row-batch-size` | 16 | 本地 Parquet 解码批大小>=1，禁用 Arrow 后台读取线程 |
| `--progress-every` | 100 | 每写入 N 条打印进度，N>=1；首条与分片开始/完成也会打印 |

默认 local 模式按固定 commit 解析所选 config/split 的文件映射，逐片用 Hub SDK 下载完整文件；HTTP 中断会重试，并在服务端支持 Range 的情况下继续 .incomplete。随后仅从本地文件解码，达到 TRAIN_LIMIT 就停止，不下载剩余不需要的分片。它会增加分片缓存磁盘占用，但避免在 Arrow 解析期间反复请求远端字节。JSON/JSONL 也支持先落盘读取；JSON 数组会整体解析到内存，Parquet/JSONL 按小批/行读取。

SOURCE_CACHE 通过 Hub SDK 的 cache_dir 显式设置，不是 HF_HOME 的别名；只改 HF_HOME 不会移动它。小样本验证仍需完整下载首个所需分片，20条样本不等于只下载20条对应的字节。分片缓存、raw JSONL 和预处理产物可能同时占盘，正式全量下载前检查可用空间。

`DOWNLOAD_MODE=stream` 是原来的远端流式路径，网络不稳定时仍可能出现 ChunkedEncodingError。Natural Questions 原始规模很大，NarrativeQA 全文也可能占用大量空间。`TRAIN_LIMIT` 是原始源前缀限制，**不是从完整数据集均匀采样**；后续预处理只在这批候选中做均匀 reservoir sampling。做最终实验时应明确报告候选池范围，或设 `TRAIN_LIMIT=0` 扫描完整训练 split。

下载每个源生成：

```text
artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw/training/naturalqa.jsonl
artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw/training/naturalqa.source.json
artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw/benchmark/naturalqa.jsonl
artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw/benchmark/naturalqa.source.json
```

`.source.json` 记录 repo/config/split、解析后的 commit、行数与文件 SHA256。相同身份且哈希匹配会复用；身份变化或不完整文件会拒绝覆盖，请换 `RAW_DIR`。不要手动修改现有 raw 文件后继续使用原 manifest。

新版另保存 transport 元数据，包括实际使用的分片文件、大小和 SHA256，保留原 identity 格式，已完成的旧版 raw 可继续复用。转换中断后可重跑同一命令：完整 JSONL 不会被半成品替换，临时 JSONL 会从头转换，已经落盘的源分片可复用；这不等于 JSONL 行级断点续写。源选择有歧义时不猜测，可在注册表该条增加 `"data_files":{"train":"default/train-*.parquet"}` 等明确映射。

## 4. 镜像与本地原始文件

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/benchmark/sh/seco/10_download_data.sh
```

若 MultiNews 转换分支在当前 endpoint 不可用，可以切回官方 endpoint；也可在 Linux 自行从合法来源准备 JSONL，然后放入 `RAW_DIR/training/multinews.jsonl`，每行包含 `document` 和 `summary`。其他源同理，文件名与 schema 必须符合注册名称。没有 `.source.json` 的本地输入会明确记为 `local_file` 加实际 SHA256，不伪造 Hub 来源。

已有 SeCo 标准数据可直接使用 `input` / `prompt` / `answer` 格式放入对应 raw 文件；answer 可以是字符串或非空字符串数组。不要将测试样本放入 training 目录。自定义适配器只需修改 `seco/data.py::normalize_record`，保留统一输出协议。

## 5. IncompleteRead / ChunkedEncodingError 修复

该错误表示 HTTP 响应未接收完整；把 timeout 增加到1200秒不能修复已断开的连接。下载入口默认改用 local 模式，默认关闭 Xet/hf_transfer，使用有 .incomplete 文件的 HTTP 下载路径；可通过环境变量显式调整。此改动不涉及模型、数据 split、抽样范围或压缩方法。

下面的传输环境默认值由 scripts/benchmark/sh/seco/10_download_data.sh 设置；直接运行 Python 时应显式 export。先停止旧下载进程，再在 Linux 项目根更新代码；有本地修改时先自行保留，git pull --ff-only 会在无法快进时停止，不覆盖本地变更。

```bash
git pull --ff-only origin main
export PYTHON_BIN=/你的环境/bin/python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# 更新代码后先做小样本验证，使用独立输出目录，避免与正式30000条身份冲突。
PROBE_DIR="artifacts/raw_probe_local_$(date +%Y%m%d_%H%M%S)"
ROLE=training TRAIN_LIMIT=20 RAW_DIR="$PROBE_DIR" DOWNLOAD_MODE=local \
bash scripts/benchmark/sh/seco/10_download_data.sh --datasets naturalqa

# 验证成功后使用原始正式配额；共享分片缓存，无需重新下载已完成分片。
ROLE=all TRAIN_LIMIT=30000 RAW_DIR=artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw DOWNLOAD_MODE=local \
bash scripts/benchmark/sh/seco/10_download_data.sh
```

探针目录仅用于验证，不会自动合并进正式 raw。正式命令显式恢复 role/limit/output，避免继承之前为探针设置的环境值。若正式路径已有同源同配额且哈希匹配的完整 JSONL，会显示 reuse；只有数据身份不变时才复用。

新流程日志按阶段显示 `[start]`、`[files]`、`[download N/M]`、`[local]`、`[rows]`、`[done]`，异常重试显示 `[retry N/M]`，完整产物复用显示 `[reuse]`。文件下载阶段观察字节进度；转换阶段观察记录数。最终 JSONL 仍由隐藏临时文件原子发布，不能因为正式文件暂未出现就认定进程无进展。

不要清空 SOURCE_CACHE 或 .incomplete 来解决普通网络中断；保留它们才能尝试续传。默认下载串行处理分片，避免多连接放大镜像不稳定问题。401/403/404、磁盘满或路径权限错误不会反复重试，应先修复对应原因。镜像文件链路仍失败时，在新进程中换成可访问的官方 endpoint，缓存和 commit 不变时可复用文件；续传是否有效取决于服务端支持。

如果改为本地读取后仍独立出现 PyGILState_Release 崩溃，保存完整首个异常，并在 Linux 执行 `python -m pip check`，记录 Python/pyarrow/datasets/huggingface_hub/requests/fsspec 版本，单独排查 native 库环境。当前堆栈只证明网络读取失败在先，不能确认解释器退出崩溃的具体库或根因。回归用例位于 `tests/benchmark/test_downloads.py`，请在目标 Linux 环境执行。

上述 local 模式和环境默认值适用于五数据集入口。NIAH 的下载设置见 [NIAH 下载指南](../niah/02_download.md)。
