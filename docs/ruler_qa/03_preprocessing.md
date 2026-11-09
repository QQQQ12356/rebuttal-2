# 03 数据预处理

```bash
bash scripts/ruler_qa/sh/20_prepare_data.sh
SPLIT=test bash scripts/ruler_qa/sh/20_prepare_data.sh
```

默认用 `configs/ruler_qa/prepare.json`。可复制为新配置，通过 `PREPARE_CONFIG` 指定副本，通过 `PREPARED_DIR` 指定新目录。每次只生成一个 split。输出包含对应的 `train.jsonl` 或 `test.jsonl` 及 `manifest.json`，整组文件就绪后发布到目标路径。

## 在脚本中调整预处理参数

在 `scripts/ruler_qa/sh/20_prepare_data.sh` 顶部修改参数。留空时从 `PREPARE_CONFIG` 读取该项。比如把 `TRAIN_LENGTHS="${TRAIN_LENGTHS:-}"` 改为 `TRAIN_LENGTHS="${TRAIN_LENGTHS:-4096 8192}"`，即可在脚本中固定训练长度。长度列表用空格分隔。

| Bash 变量 | Python 参数 | 默认配置与作用 |
| --- | --- | --- |
| `TRAIN_LENGTHS` | `--train-lengths` | `4096 8192`，训练 token 档位，每题只分到一个档位。 |
| `TEST_LENGTHS` | `--test-lengths` | `4096 8192 16384 32768`，每道入选测试题覆盖全部档位。 |
| `TRAIN_CANDIDATES_PER_TASK` | `--train-candidates-per-task` | `4096`，每任务候选池大小，至少等于该任务训练题量。 |
| `TRAIN_SAMPLES_PER_TASK` | `--train-samples-per-task` | `512`，每任务接受的训练题量，至少覆盖所有训练档位。 |
| `TEST_SAMPLES_PER_TASK` | `--test-samples-per-task` | `50`，每任务接受的测试题量，每题按测试档位数展开。 |
| `MIN_FILL` | `--min-fill` | `0.85`，目标长度的最低填充比例，范围为 `(0, 1]`。 |
| `MIN_RATIO` | `--min-ratio` | `4`，用于检查 decoder 预算的最低压缩比，至少为 1。 |
| `MAX_DECODER_TOKENS` | `--max-decoder-tokens` | `16384`，压缩后上下文、提问后缀及答案或生成预留的总预算。 |
| `MAX_ANSWER_TOKENS` | `--max-answer-tokens` | `64`，训练答案上限，包含 EOS。 |
| `GENERATION_BUDGET` | `--generation-budget` | `32`，测试预留的生成 token 数。测试档位须大于该值。 |
| `SEED` | `--seed` | `42`，控制问题抽样和文档选择、排列。 |

从低到高的优先级依次是 JSON 配置、Bash 脚本中的默认值、运行时的环境变量、命令末尾追加的 CLI 参数。直接调用 Python 时，CLI 中显式填写的值覆盖 JSON 中对应的项。合并后的值写入 `manifest.json` 的 `arguments`。

例如，在 Linux 上生成每任务 20 题的 4K/8K 测试集：

```bash
SPLIT=test TEST_LENGTHS='4096 8192' TEST_SAMPLES_PER_TASK=20 \
  PREPARED_DIR=artifacts/ruler_qa/prepared/test_small \
  bash scripts/ruler_qa/sh/20_prepare_data.sh
```

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
