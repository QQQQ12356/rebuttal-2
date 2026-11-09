# 03 数据预处理

```bash
bash scripts/ruler_qa/sh/20_prepare_data.sh
SPLIT=test bash scripts/ruler_qa/sh/20_prepare_data.sh
```

默认用 `configs/ruler_qa/prepare.json`。可复制为新配置，通过 `PREPARE_CONFIG` 指定副本，通过 `PREPARED_DIR` 指定新目录。每次只生成一个 split。输出包含对应的 `train.jsonl` 或 `test.jsonl` 及 `manifest.json`，整组文件就绪后发布到目标路径。

## 输入与提问后缀

默认使用 `raw`。上下文由带 `Document:` 分隔标记的文档组成，开头省略英文作答指令。后缀只包含一个用于分隔的换行和原始问题。以下是最后一篇文档到输入末尾的示意。

```text
Document:
[文档正文]
Who first introduced social insurance programs?
```

预处理分别编码上下文和后缀，再拼成 `input_ids`。上下文的 `prompt_mask` 为 0，后缀的 mask 为 1。答案及 EOS 仍单独写入训练 `labels`。测试从问题末尾开始生成。

| Bash 变量 | Python 参数 | 取值与修改方式 |
| --- | --- | --- |
| `PROMPT_STYLE` | `--prompt-style` | 默认 `raw`，使用上述后缀。设为 `legacy` 可复现此前 RULER QA 的英文指令和 `\n\nQuestion: {question}\nAnswer:` 后缀。 |
| `SPLIT` | `--split` | `train` 或 `test`。每次生成一个数据划分。 |
| `PREPARED_DIR` | `--output` | 指定一个尚不存在的目录，避免覆盖旧数据。 |
| `PREPARE_CONFIG` | `--config` | 指定长度档位、样本数和 token 预算等配置。 |

长期修改可编辑 `scripts/ruler_qa/sh/20_prepare_data.sh` 顶部的 `PROMPT_STYLE` 默认值。单次运行可在命令前设置同名环境变量，或在脚本名后追加 `--prompt-style raw`。同时指定时以末尾的 CLI 参数为准。实际格式写入 `manifest.json` 的 `arguments.prompt_style`。

```bash
SPLIT=train PROMPT_STYLE=raw PREPARED_DIR=artifacts/ruler_qa/prepared/train_raw \
  bash scripts/ruler_qa/sh/20_prepare_data.sh
SPLIT=test PROMPT_STYLE=raw PREPARED_DIR=artifacts/ruler_qa/prepared/test_raw \
  bash scripts/ruler_qa/sh/20_prepare_data.sh
```

旧 JSONL 已存储 token IDs，修改格式参数后须重新预处理。训练时将 `PREPARED_DIR` 指向 `train_raw`，测试时将 `TEST_DIR` 指向 `test_raw`。若要评估新格式下训练和测试的完整流程，使用新训练数据训练，再用对应的新测试数据评测。与旧格式做对照时核对样本 ID，因为模板长度变化可能改变能装入预算的文档和入选问题。

## 样本与预算

| 阶段 | 样本与预算 |
| --- | --- |
| 训练 | 各任务最多先抽 4096 个训练候选，再选各 512 题。合格题轮流分到 4096 和 8192 预算，每题只出现在一个长度 |
| 测试 | 每任务 50 题，均需通过 4096、8192、16384 和 32768 四个预算。同一任务每个长度复用同一批题，共 400 行 |
| 编码预算 | 训练的目标长度包含编码器输入和训练答案（含 EOS）。测试目标长度包含编码器输入和 32 个预留生成 token |
| 解码预算 | 在 `min_ratio` 下检查压缩后上下文、问题和答案或生成预留的总长度 |

流程按固定随机种子抽样。训练过滤两个 dev 来源中大小写和空白归一化后重复的问题，并在训练任务之间去重。`qa_1` 保留答案所在段落，优先加同文章段落，随后从同 split 文档池取干扰段落。`qa_2` 保留原样本的全部上下文段落，再添加同 split 的干扰文档。文档次序可复现，所有保留文档保持完整。不同 split 的原始维基段落可能重复，去重单位是问题文本。

文档、提示和训练标签分别编码，`prompt_mask` 区分可压缩上下文和完整保留的问题后缀。筛掉证据本身超长、训练答案过长、压缩后解码超预算或长度不足 `min_fill` 的候选。测试候选在任一长度失败就换题。达到配额前候选用完会报错并显示拒绝原因。

清单记录设置、模型与 tokenizer 身份、源文件及 JSONL 的 SHA-256、各任务各长度行数、拒绝统计和样本身份。训练清单另存问题哈希，用于评测检查 train/dev 问题重叠。

| 调整目的 | 修改项 |
| --- | --- |
| 降低显存与耗时 | 减小 `train_lengths` 或 `test_lengths`，也可减小样本配额 |
| 增大训练样本量 | 同时提高 `train_samples_per_task` 和 `train_candidates_per_task`，保留足够候选余量 |
| 使用更小的压缩比 | 降低 `min_ratio` 并为解码器增加预算或缩短目标长度，然后重做对应 split |
| 解决 `low_fill` | 检查候选池和解码预算，必要时减小 `min_fill` 或目标长度 |
| 生成更长答案 | 提高测试 `generation_budget` 后重做测试网格，评测时同步调整 `MAX_NEW_TOKENS` |

完整参数见 [06 配置与脚本变量](06_parameters.md)。
