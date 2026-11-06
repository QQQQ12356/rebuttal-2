# 03 数据预处理与样本筛选

默认先扫描四个原始 Twwilght benchmark 建立重复排除集合，再预处理训练源。测试抽样、训练抽样、长度策略、tokenizer 指纹与 raw/prepared 文件哈希统一保存到 `manifest.json`。数据不做静默截断。

## 1. 使用方式

```bash
bash scripts/sh/20_prepare_data.sh

# 新数据版本：更严格的输入长度；不能覆盖已有 prepared 目录。
PREPARED_DIR=artifacts/prepared/L8192_s42 MAX_INPUT_TOKENS=8192 \
    bash scripts/sh/20_prepare_data.sh

# 仅选择两个训练源；测试仍处理全部四个 Twwilght 源。
python scripts/py/20_prepare_data.py \
    --model artifacts/models/Qwen3-4B-Instruct-2507 \
    --output artifacts/prepared/two_sources \
    --datasets naturalqa hotpotqa --train-samples 2000 3000
```

Shell 顶部的 `DATASETS` 和 `TRAIN_SAMPLES` 是数组，按位置一一对应；也可向 Python 只传一个训练数，所有训练源使用该数。`0` 表示使用全部合格样本，训练数据不是不经筛选的前 N 条。

## 2. 筛选顺序

1. 适配源 schema，构造标准 context、prompt、参考答案与稳定 ID。
2. Natural Questions 去掉 HTML token；同一标注中的多个 short spans 拼成一个完整答案，不把不同 span 当成答案别名；不可回答样本过滤。
3. 训练源与全部可正常适配的 raw benchmark 比较：完整 context 去空白归一化后精确匹配则排除；非摘要任务还比较同数据集内归一化问题。使用全部 raw 测试行建立集合，不只使用抽中的 500 条。
4. 同一个源内按 context+prompt 精确去重，保留同一文章的不同问题。MultiNews 的恒定摘要指令不参与问题去污染，避免误删全部新闻训练样本。
5. context、prompt 分别编码后拼接，answer 首个参考答案单独编码并追加 EOS。
6. 按输入、答案、解码序列预算筛选；不截断 context、prompt 或 answer。
7. 对所有合格候选做固定种子的 reservoir sampling，再打乱样本顺序；所有训练源合并成一个训练集。

去污染仅覆盖上述精确文本/归一化问题规则，不等同于语义去重，也不能保证不存在所有近似重复。上下文 title 与源顺序保留，`isgold`、supporting facts 和 reference answer 不参与输入筛选。MultiNews 的原始多新闻分隔符也保留。

## 3. 参数表

| Python 参数 | 默认 | 作用与修改提示 |
| --- | --- | --- |
| `--raw-dir` | artifacts/raw | raw 根目录，须有四个 benchmark 文件以及选中的训练源 |
| `--output` | artifacts/prepared/default | 新的 prepared 根目录；已有目录报错 |
| `--model` / `--revision` | 本地默认模型目录 / main | 用于分词的模型/tokenizer；远程版本建议写 commit |
| `--datasets` | 五个训练源 | naturalqa / 2wikimqa / hotpotqa / narrativeqa / multinews |
| `--train-samples` | 每源 5000 | 筛选后目标数量；单数或与 datasets 等长数组；0 全部 |
| `--test-samples` | 500 | 每个 benchmark 的合格目标数量；0 全部 |
| `--candidate-limit` | 0 | 每个训练文件最多扫描前 N 行；0 扫描已下载全部行 |
| `--max-input-tokens` | 16384 | context+prompt 上限，含完整 prompt |
| `--max-answer-tokens` | 1024 | 仅训练 answer+EOS 上限；不依据测试参考答案长度过滤 |
| `--max-decoder-tokens` | 8192 | 压缩 context+完整 prompt+训练 answer 或测试生成预留上限 |
| `--min-context-tokens` | 32 | 输入正文最小 token 数 |
| `--min-ratio` | 4 | 计划使用的最小训练/测试压缩比；决定最坏的中心数量 |
| `--test-generation-budget` | 512 | 测试的生成 token 预留；必须覆盖最大 max_new_tokens |
| `--seed` | 42 | 抽样、数据源独立 RNG 和全局混合 shuffle 的种子 |
| `--strict-count` | 关闭 | 开启后，任一数据源不足目标样本数就失败 |

长度约束为：`L_context + L_prompt <= max_input_tokens`，`K + L_prompt + L_answer <= max_decoder_tokens`，其中 `K=min(L_context,max(2,ceil(L_context/min_ratio)))`。测试将 `L_answer` 替换为 generation budget。实际模型位置上限还会在训练/测试入口检查。

压缩比从 `[4,8,16,32]` 改为 `[2,4,8]` 时，必须至少以 `MIN_RATIO=2` 重新预处理。更换 tokenizer、扩大输入长度、扩大生成预算也应生成新的 prepared 目录。只减少测试样本数或只测试已有比例数组的子集不需要重新预处理。

## 4. 产物与统计

```text
artifacts/prepared/default/
  manifest.json
  train.jsonl               混合训练样本
  train/<dataset>.jsonl     各源已筛选样本
  test/<dataset>.jsonl      四个独立测试源
```

标准行同时包含原始 `input/prompt/answer`、来源 ID，以及 `input_ids/prompt_mask/labels` 和长度。`prompt_mask=1` 仅指完整 prompt，padding 不属于 prompt；训练与测试读取同一套已保存 token 边界。训练只监督第一个参考答案，EM/F1 针对全部参考答案取最高值。

`manifest.statistics` 中的计数包括 `scanned`、`invalid_or_unanswerable`、`test_overlap`、`duplicates`、`short_context`、`long_input`、`long_answer`、`long_decoder`、`eligible`、`selected`；零值计数可能不出现。NarrativeQA 的全文较长，默认长度下可能远不足 5000 条，应检查统计后增加候选池/长度或接受较少样本，不要直接换成人工摘要。全部不合格会明确报错。

预处理失败时可能保留未完成目录；缺少完成 manifest 的目录不能用于训练。保留其日志以检查原因，然后更换 `PREPARED_DIR` 重试。
