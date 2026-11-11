# 02 数据下载与预处理

先设置方法，并完成 [01 环境与模型下载](01_environment.md)。

```bash
method=comi
bash "scripts/benchmark/sh/$method/10_download_data.sh"
bash "scripts/benchmark/sh/$method/20_prepare_data.sh"

bash "scripts/niah/sh/$method/10_download_data.sh"
bash "scripts/niah/sh/$method/20_prepare_train.sh"
bash "scripts/niah/sh/$method/21_prepare_grid.sh"

bash "scripts/ruler_qa/sh/$method/10_download_data.sh"
SPLIT=train bash "scripts/ruler_qa/sh/$method/20_prepare_data.sh"
SPLIT=test bash "scripts/ruler_qa/sh/$method/20_prepare_data.sh"
```

| 任务 | 下载配置 | 预处理配置 | 默认输出目录 |
| --- | --- | --- | --- |
| Benchmark | `configs/benchmark/datasets.json` | `20_prepare_data.sh` 中的长度、配额和采样参数 | `prepared/default` |
| NIAH | `configs/niah/sources.json` | `prepare_train.json` 管训练，`grid.json` 管测试 | `prepared/train_v1`、`prepared/grid_v1` |
| RULER QA | `configs/ruler_qa/sources.json` | `prepare.json` 管长度、候选数、填充比例与采样 | `prepared/train_v1`、`prepared/test_v1` |

- `RAW_DIR` 指向下载输出。下载完成后可供不同方法重复使用。
- `PREPARED_DIR` 指向新的预处理目录。NIAH 网格用 `GRID_DIR`。预处理清单记录 tokenizer 指纹、数据文件哈希和筛选配置。
- `MODEL_PATH` 必须匹配后续训练或评测的 tokenizer。基座一致时，各方法可显式引用同一份 prepared 数据以固定样本与顺序。
- `min_ratio` 约束后续最低训练和评测比例。预处理用名义压缩预算筛样本。评测会根据方法参数检查额外记忆长度，RAM 保留原文较多时可能需要提高 `max_decoder_tokens`、缩短输入或降低 `keep_ratio`。
- `generation_budget` 为 NIAH/RULER QA 预留生成长度。Benchmark 用 `TEST_GENERATION_BUDGET`。生成脚本的 `MAX_NEW_TOKENS` 或任务级生成上限必须在预算内。
- NIAH 的训练与测试背景及答案做去重，RULER QA 的训练与测试按问题身份检查重叠。复用已有数据时保留清单和对应的 JSONL 文件。

下载源、筛选条件、网格定义和每个变量的完整说明分别见 [Benchmark](benchmark/03_preprocessing.md)、[NIAH 训练数据](niah/03_prepare_train.md)、[NIAH 网格](niah/05_prepare_grid.md) 和 [RULER QA](ruler_qa/03_preprocessing.md)。
