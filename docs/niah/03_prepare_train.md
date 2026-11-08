# 03 NIAH 训练数据预处理

独立入口只处理 `configs/niah/prepare_train.json` 声明的源，不读取 NaturalQA 等数据。默认目标为 synthetic 4000 + Cogito Bits 2000 + Cogito Bind 2000。候选不足时记录实际数并警告，可用 strict_count 强制配额。

```bash
bash scripts/niah/sh/20_prepare_train.sh

# 快速检查样本：仍只在 Linux 执行，并使用新输出目录。
PREPARED_DIR=artifacts/niah/prepared/train_small \
SYNTHETIC_SAMPLES=32 SUPERVISED_SAMPLES=16 \
    bash scripts/niah/sh/20_prepare_train.sh
```

## 1. 处理规则

1. 读取 heldout_corpora 的完整正文哈希集合，排除训练背景中的精确归一化重复文档。
2. synthetic 从 PG19 train 的已分词正文池随机取窗口，在随机深度插入新 passkey/UUID，长度从数组抽取；针的 key 和答案不出现在问题中，答案只能从背景检索。
3. supervised 读取两个源的 context/query/answer 文本，丢弃其他 tokenizer 的 IDs，使用本次 Qwen tokenizer 重新编码。
4. 按 context/prompt 内容和 ID 去重，按输入、答案与压缩后解码预算过滤；已经含针的样本不截尾，避免删掉监督证据。
5. 对每源合格候选做独立固定种子的 reservoir sampling，再全局打乱，保存 train.jsonl 和完整 manifest。

合成背景可在插针前裁出窗口，这是任务构造；监督样本则超长过滤。二者不能混同为“先插针再截断”。数据序列边界与现有 SeCo 一致，context 与 prompt 分开处理；不额外套聊天模板。训练使用第一个参考答案加 EOS 进行 teacher forcing。

## 2. 配置表

| prepare_train.json 路径 | 默认 | 说明 |
| --- | --- | --- |
| seed | 42 | 采样与合成 RNG；训练和测试使用不同 namespace |
| strict_count | false | true 时任一源不足目标数报错 |
| candidate_limit | 0 | 每个监督源扫描前 N 条 raw；0 扫描全部已下载行 |
| heldout_corpora | [paulgraham] | 必须存在的测试背景，训练去污染时读取 |
| limits.max_input_tokens | 16384 | context+完整 prompt 最大长度 |
| limits.max_answer_tokens | 1024 | 训练 answer+EOS 上限 |
| limits.max_decoder_tokens | 8192 | centers+完整 prompt+answer 的上限 |
| limits.min_context_tokens | 32 | 最短正文 |
| limits.min_ratio | 4 | 计划使用的最小压缩比，决定最大中心数 |
| synthetic.samples | 4000 | 合成目标数；0 禁用合成源 |
| synthetic.candidates | 8000 | 先生成多少候选，再筛选和采样；不是最终样本数 |
| synthetic.corpus | pg19_train | 训练背景文件名，不含 .jsonl |
| synthetic.lengths | [2048,4096,8192,16384] | 完整 encoder token 长度候选，均匀抽取 |
| synthetic.tasks | [uuid,passkey] | 每条随机选择任务；训练不允许固定 custom 针 |
| synthetic.max_corpus_tokens | 2000000 | 背景池 token 上限，控制 RAM |
| synthetic.per_document_tokens | 65536 | 每篇最多取这些正文 token，控制单篇占比 |
| synthetic.repeat_haystack | false | 默认背景不循环重复；true 仅用于显式重复实验 |
| supervised[].source | cogito_bits / cogito_bind | 每个监督源的 raw 文件名 |
| supervised[].samples | 各 2000 | 合格样本目标，0 全部；删掉数组项可禁用该源 |

`K=min(L_context,max(2,ceil(L_context/min_ratio)))`。筛选要求 `L_context+L_prompt<=max_input_tokens`、`L_answer<=max_answer_tokens`、`K+L_prompt+L_answer<=max_decoder_tokens`。源数据的 seq_len 字段来自其他 tokenizer，只作原始元数据，不能代替 Qwen 重新计算的长度。

Python 入口参数：`--config` 选择上述配置；`--raw-dir/--output` 选独立目录；`--model/--revision` 选 tokenizer；`--synthetic-samples` 覆盖合成目标数，0 禁用；`--supervised-samples` 覆盖每个监督源的配额，0 全部合格。Shell 中相应变量说明见 [07](07_parameters.md)。

## 3. 产物与调整

输出目录含 `train.jsonl` 和 `manifest.json`。manifest 为 `format=seco-data-v1, task=niah, kind=train`，记录源哈希/commit、训练背景文档身份、已选答案哈希、tokenizer 指纹、配置与各源统计。只有完成 manifest 的产物可训练。默认输出不可覆盖；改变配置后使用新 PREPARED_DIR。

保留 checkpoint 对应的这两个文件，以便后续按 [09 训练集评测](09_train_benchmark.md) 在原样本上生成并计分。该入口会核对 checkpoint 保存的 manifest SHA-256，以及 manifest 记录的 `train.jsonl` SHA-256。

统计含 scanned、eligible、selected、duplicates、heldout_overlap、long_input、long_answer、long_decoder 等；零值可不出现。没有合格样本会报错。默认长度下较长的 Cogito 样本可能因 Qwen token 数超过上限被过滤，不能直接承诺每源最终 2000 条。

扩大样本数时同时扩大 raw 候选池和 synthetic.candidates。降低压缩比下限时同步降低 limits.min_ratio，并重新预处理。增大长度时同时核对训练 JSON、解码预算、模型位置上限与实际显存。缩小背景池过度可能造成背景窗口重复，应在论文记录池大小、每篇上限、源前缀和种子。

去污染是精确归一化文档/输入匹配，加上训练/测试答案 namespace 与可选答案哈希检查；不等同于语义去重，也不保证消除所有近似重复。
