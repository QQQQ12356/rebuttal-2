# SeCo：Linux 训练、压缩推理与 EM/F1 benchmark

新增独立大海捞针流程，见 [NIAH 实验指南](docs/niah/00_README.md)：脚本位于 `scripts/niah/py` 与 `scripts/niah/sh`，配置/数据/run/热力图与本页五数据集流程分开。

本目录以用户放入的原始 SeCo 为基线，保留两个独立语言模型、两套 LoRA，以及完整的语义中心选择、聚类分配和加权压缩方法。默认基座为 `Qwen/Qwen3-4B-Instruct-2507`，即官方 Qwen3 4B Instruct 非思考版本。

**当前 Windows 仅修改代码和做静态检查。模型下载、数据下载、预处理、训练、推理、单元测试与 Excel 导出均在 Linux 运行。** 本项目没有宣称完成真实 GPU 实验或测得加速倍数。

| 顺序 | 指南 | 内容 |
| --- | --- | --- |
| 01 | [环境](docs/01_environment.md) | Python/CUDA、固定依赖、DeepSpeed 与离线运行 |
| 02 | [模型与数据下载](docs/02_download.md) | 五个训练源、四个 Twwilght 测试源、镜像与本地输入 |
| 03 | [预处理](docs/03_preprocessing.md) | 样本数量、长度筛选、去重、去污染、token 边界 |
| 04 | [训练](docs/04_training.md) | 单卡/DDP/ZeRO、随机压缩比、断点恢复、参数表 |
| 05 | [推理与打分](docs/05_benchmark.md) | 合并 LoRA、多卡分片、固定比例扫描、EM/F1 |
| 06 | [Excel](docs/06_excel.md) | 汇总指标、逐样本内容、旧版结果导入 |
| 07 | [架构与方法保持](docs/07_architecture.md) | 原始算法、加速来源、代码规范与文件职责 |
| 08 | [Shell 变量索引](docs/08_script_variables.md) | 每个入口可修改的变量、优先级与示例 |
| 09 | [验证与排错](docs/09_validation.md) | 静态检查记录、Linux CPU/GPU 检查流程 |

文件组织如下；Python 与 Shell 分目录存放，代码不依赖相邻工程。

```text
seco/                  模型、原始压缩方法、数据适配、指标、训练保存、Excel 导出
scripts/py/            编号 Python 命令；均支持 --help
scripts/sh/            同编号 Bash 启动入口；顶部提供变量与注释
configs/               数据来源、训练配置、ZeRO 配置
tests/                 Linux 验证用例；reference_compression.py 为冻结基线
docs/                  编号实验指南
legacy/py/             原始 Python 代码，包含原始 test_fast 的 Python 文件
legacy/sh/             原始 Shell 脚本，包含原始 test_fast 的 Shell 文件
artifacts/             Linux 运行后生成：models/raw/prepared/runs/benchmarks/reports
logs/                  Linux 训练/评测日志
```

Linux 快速路径：先激活环境并进入本项目目录，再依次执行。实际参数先按指南确认；尤其是 GPU 编号、样本数、模型路径和显存预算。

```bash
bash scripts/sh/00_setup_environment.sh
bash scripts/sh/01_download_model.sh
bash scripts/sh/10_download_data.sh
bash scripts/sh/20_prepare_data.sh
PLAN_ONLY=1 bash scripts/sh/30_train.sh
bash scripts/sh/30_train.sh

# 将下面的路径替换为训练日志打印的真实目录。
export CHECKPOINT=artifacts/runs/你的训练目录/final
bash scripts/sh/40_merge_lora.sh
CHECKPOINT="${CHECKPOINT}_merged" bash scripts/sh/50_benchmark.sh
bash scripts/sh/70_export_excel.sh
```

默认训练每个源目标采样 5,000 条，五源共至多 25,000 条；不足时警告并使用全部合格样本，实际数量以 `manifest.json` 为准。默认测试每个 Twwilght 源 500 条、扫描压缩比 `[4,8,16,32]`。NarrativeQA 提供训练与数据适配，测试入口暂不启用。

所有新训练、预处理、benchmark 和 Excel 路径遵守不覆盖已有产物的约束。旧脚本保存在 `legacy/` 用于核对；其路径已归档，不能作为新实验入口直接使用。旧 checkpoint 的转换见 [05 推理指南](docs/05_benchmark.md)。

五数据集下载入口 `scripts/sh/10_download_data.sh` 现默认 `DOWNLOAD_MODE=local`：先把固定 commit 的源分片下载到 `artifacts/source_cache/`，再从本地解码并输出 raw JSONL；支持截断响应重试、可用时的分片续传以及分阶段/行数进度。达到训练样本上限后不再下载后续分片。`DOWNLOAD_MODE=stream` 可显式使用旧远端读取路径。

若遇到 `IncompleteRead`、`ChunkedEncodingError` 或解析文件列表后长时间无输出，请先按 [02 下载指南第5节](docs/02_download.md#5-incompleteread--chunkedencodingerror-修复) 更新代码并做20条样本验证。分片缓存应保留，raw JSONL 转换中断则从头转换；两种恢复粒度不同。此下载模式属于五数据集入口，NIAH 独立下载入口的当前行为见 [NIAH 下载指南](docs/niah/02_download.md)。

五数据集下载入口 `scripts/sh/10_download_data.sh` 现默认 `DOWNLOAD_MODE=local`：先把固定 commit 的源分片下载到 `artifacts/source_cache/`，再从本地解码并输出 raw JSONL；支持截断响应重试、可用时的分片续传以及分阶段/行数进度。达到训练样本上限后不再下载后续分片。`DOWNLOAD_MODE=stream` 可显式使用旧远端读取路径。

若遇到 `IncompleteRead`、`ChunkedEncodingError` 或解析文件列表后长时间无输出，请先按 [02 下载指南第5节](docs/02_download.md#5-incompleteread--chunkedencodingerror-修复) 更新代码并做20条样本验证。分片缓存应保留，raw JSONL 转换中断则从头转换；两种恢复粒度不同。此下载模式属于五数据集入口，NIAH 独立下载入口的当前行为见 [NIAH 下载指南](docs/niah/02_download.md)。
