# 10 自然语言短答案 NIAH

## 1. 样本格式

| 字段 | 示例 | 用途 |
| --- | --- | --- |
| 问题 | Which place does the harbor visitor center recommend visiting after lunch? | 放在长文本末尾，指定推荐机构与时间 |
| 针 | The harbor visitor center recommends visiting the maritime museum after lunch. | 完整事实句，插入背景的目标位置 |
| 答案 | the maritime museum | 针中的连续地点短语，作为训练标签和评测参考答案 |

输入组装、插针深度和长度定义见 [05 评测网格](05_prepare_grid.md)。训练与评测共享句式。机构和时间由种子抽取，地点从各自的词表中抽取。默认两个地点词表没有交集，每个词表包含 8 个短语。地点可在同一集合中重复，适合用同一规律构造多条检索样本。扩大事实覆盖时，直接编辑 `seco/niah/data.py` 的 `CUSTOM_DESTINATIONS`，保持两份词表互斥，优先用在背景中较少出现的完整地点短语。修改问题时只提供定位线索，保留答案在针中的连续原文。

## 2. 参数与覆盖顺序

| 用途 | Python 参数 | Shell 变量 | JSON 字段和默认行为 |
| --- | --- | --- | --- |
| 选择合成训练任务 | `--synthetic-tasks custom` | `SYNTHETIC_TASKS` | `synthetic.tasks`，默认 `["uuid","passkey","custom"]` |
| 设置合成训练目标条数 | `--synthetic-samples 4000` | `SYNTHETIC_SAMPLES` | `synthetic.samples`，默认 4000；0 禁用合成训练 |
| 选择评测任务 | `--task custom` | `GRID_TASK` | `task`，16K 网格默认 `custom`，128K 配置默认 `uuid` |
| 选择训练配置 | `--config 路径` | `PREPARE_CONFIG` | 默认 `configs/niah/prepare_train.json` |
| 选择评测配置 | `--config 路径` | `GRID_CONFIG` | 默认 `configs/niah/grid.json` |
| 设置输出目录 | `--output 路径` | 训练用 `PREPARED_DIR`；评测用 `GRID_DIR` | 选新目录，避免与旧产物混用 |
| 指定训练清单 | `--training-manifest 路径` | `TRAINING_MANIFEST` | 为评测核对训练答案与背景；自定义训练目录时显式传入 |
| 选择原始数据及 tokenizer | `--raw-dir 路径 --model 路径` | `RAW_DIR`、`MODEL_PATH` | 训练和评测使用与模型一致的 tokenizer |

Shell 文件顶部的变量可直接修改，也可以在调用前临时赋值。空的任务变量沿用 JSON。显式的 Python 命令行参数覆盖 JSON。多个训练任务用空格分隔，例如 `SYNTHETIC_TASKS="uuid passkey custom"`，直接调用 Python 时写成 `--synthetic-tasks uuid passkey custom`。

训练 JSON 中的 `synthetic.candidates` 控制候选池大小，`synthetic.lengths` 控制输入长度选项，`seed` 控制抽样与针内容。训练任务列表只控制合成来源。若要导出全为自然语言短答案的训练集，复制默认 JSON 到新文件，设置 `synthetic.tasks` 为 `["custom"]` 且 `supervised` 为 `[]`，再通过 `PREPARE_CONFIG` 选择新配置。监督配额设为 0 会保留全部合格样本。

## 3. Linux 调用示例

从仓库根目录运行。以下命令保留默认监督来源，仅将合成训练样本切换到 custom。

```bash
PREPARED_DIR=artifacts/niah/prepared/train_custom_v1 SYNTHETIC_TASKS=custom \
    bash scripts/niah/sh/20_prepare_train.sh

GRID_DIR=artifacts/niah/prepared/grid_custom_v1 GRID_TASK=custom \
    TRAINING_MANIFEST=artifacts/niah/prepared/train_custom_v1/manifest.json \
    bash scripts/niah/sh/21_prepare_grid.sh
```

也可直接传 Python 参数：

```bash
python scripts/niah/py/20_prepare_train.py --synthetic-tasks custom \
    --output artifacts/niah/prepared/train_custom_v1

python scripts/niah/py/21_prepare_grid.py --task custom \
    --output artifacts/niah/prepared/grid_custom_v1 \
    --training-manifest artifacts/niah/prepared/train_custom_v1/manifest.json
```

两组命令任选一组。下游训练和评测分别传入新 `PREPARED_DIR` 和 `GRID_DIR`，参见 [04 训练](04_training.md) 和 [06 评测与绘图](06_benchmark_plot.md)。

## 4. 固定事实评测与排查

- 自动生成模式下，网格 JSON 设 `"task": "custom"` 并省略 `custom` 对象。同一个 trial 在所有长度和深度使用同一条事实，跨 trial 按种子抽取，可能抽到相同的地点。
- 固定事实模式下，在网格 JSON 顶层加入 `custom` 对象，填写 `needle`、`question` 和 `answer`。该三元组用于所有 trial。问题应能定位针，答案必须是针中的原文。字段示例见 [05 评测网格](05_prepare_grid.md)。
- 背景已经含有答案时，生成器会报错。可换评测种子或选更有区分度的答案短语。训练清单报告答案重叠时，固定事实应换用训练集未出现的答案；编辑自动词表时应恢复两集合的答案互斥。
- 答案长度必须在生成预算内。修改 `generation_budget` 时同步核对评测脚本的 `MAX_NEW_TOKENS` 和解码器预算。已有输出目录需换新路径再运行。
