# 01 环境与预训练数据

在 Linux 项目根目录激活 Python 3.10 或 3.11 环境。完成 Llama 模型访问授权后配置 Hugging Face 凭据。[Linux 环境与模型下载](../01_environment.md)。

```bash
INSTALL_DEEPSPEED=0 bash scripts/benchmark/sh/icae/00_setup_environment.sh
bash scripts/benchmark/sh/icae/01_download_model.sh
```

预训练接受本地 UTF-8 JSONL 或 Parquet 分片。默认读取字符串列 `text`。每行是一篇文档。可自行准备语料，或从有访问权限的 Hub 数据集下载分片后传入脚本。跨实验复用语料时固定分片、输入顺序和 revision。

```bash
raw=artifacts/icae/Llama-3.2-1B-Instruct/icae_pretrain/raw
mkdir -p "$raw"
# 使用 Hub 语料时，替换仓库、commit 和分片匹配模式
hf download org/dataset --type dataset --revision your_commit_sha \
    --include 'data/train-*.parquet' --local-dir "$raw"

# 将语料放入 "$raw/documents.jsonl"，每行形如 {"text":"文档全文"}
CORPUS_FILE="$raw/documents.jsonl" bash scripts/icae/sh/20_prepare_pretrain.sh

# 多个分片，按给定顺序读取
PREPARED_DIR=artifacts/icae/Llama-3.2-1B-Instruct/icae_pretrain/prepared/v2 \
    bash scripts/icae/sh/20_prepare_pretrain.sh --input "$raw"/data/train-*.parquet
```

模型 tokenizer 对文档分词，先按原始 token 数筛选，再按中点切分。对 token `[a,b,c,d,e]`，编码器输入是 `[BOS,a,b]`，AE 目标是 `[BOS,a,b,EOS]`，LM 目标是 `[c,d,e,EOS]`。长度超限的文档被过滤。预处理按已分词的前半段去重，前 `EVAL_SAMPLES` 条用于验证，随后 `TRAIN_SAMPLES` 条用于训练。语料不足时会报告缺少的 split，换用更多分片或降低配额。

输出目录含 `train.jsonl`、可选的 `eval.jsonl` 和 `manifest.json`。清单记录原始分片哈希、目标文件哈希、各列最大长度、实际数量和 tokenizer 指纹。更换语料、tokenizer 或配额时指定新 `PREPARED_DIR`。

小规模检查可设置 `TRAIN_SAMPLES=8 EVAL_SAMPLES=2 MIN_TOKENS=32 MAX_TOKENS=256`，并为预处理与预训练设置同一个独立的 `PREPARED_DIR`。
