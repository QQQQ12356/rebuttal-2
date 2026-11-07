# 00 Benchmark 实验入口

本目录以用户放入的原始 SeCo 为基线，保留两个独立语言模型、两套 LoRA，以及完整的语义中心选择、聚类分配和加权压缩方法。默认基座为 `Qwen/Qwen3-4B-Instruct-2507`，即官方 Qwen3 4B Instruct 非思考版本。

**当前 Windows 仅修改代码和做静态检查。模型下载、数据下载、预处理、训练、推理、单元测试与 Excel 导出均在 Linux 运行。** 本项目没有宣称完成真实 GPU 实验或测得加速倍数。

| 顺序 | 指南 | 内容 |
| --- | --- | --- |
| 01 | [环境](01_environment.md) | Python/CUDA、固定依赖、DeepSpeed 与离线运行 |
| 02 | [模型与数据下载](02_download.md) | 五个训练源、四个 Twwilght 测试源、镜像与本地输入 |
| 03 | [预处理](03_preprocessing.md) | 样本数量、长度筛选、去重、去污染、token 边界 |
| 04 | [训练](04_training.md) | 单卡/DDP/ZeRO、随机压缩比、断点恢复、参数表 |
| 05 | [推理与打分](05_benchmark.md) | 合并 LoRA、多卡分片、固定比例扫描、EM/F1 |
| 06 | [Excel](06_excel.md) | 汇总指标、逐样本内容、旧版结果导入 |
| 07 | [架构与方法保持](07_architecture.md) | 原始算法、加速来源、代码规范与文件职责 |
| 08 | [Shell 变量索引](08_script_variables.md) | 每个入口可修改的变量、优先级与示例 |
| 09 | [验证与排错](09_validation.md) | 静态检查记录、Linux CPU/GPU 检查流程 |

```text
configs/benchmark/      数据注册表和训练配置
scripts/benchmark/py/   Python 命令
scripts/benchmark/sh/   Bash 启动入口与参数默认值
tests/benchmark/        数据、下载、指标和 Excel 用例
artifacts/              models、raw、prepared、runs、benchmarks、reports
logs/                   训练和评测日志
```

在 Linux 上激活环境并进入项目根目录，再按指南确认 GPU 编号、样本数、模型路径和显存预算。

```bash
bash scripts/benchmark/sh/00_setup_environment.sh
bash scripts/benchmark/sh/01_download_model.sh
bash scripts/benchmark/sh/10_download_data.sh
bash scripts/benchmark/sh/20_prepare_data.sh
PLAN_ONLY=1 bash scripts/benchmark/sh/30_train.sh
bash scripts/benchmark/sh/30_train.sh

# 将下面的路径替换为训练日志打印的真实目录。
export CHECKPOINT=artifacts/runs/你的训练目录/final
bash scripts/benchmark/sh/40_merge_lora.sh
CHECKPOINT="${CHECKPOINT}_merged" bash scripts/benchmark/sh/50_benchmark.sh
bash scripts/benchmark/sh/70_export_excel.sh
```

默认训练每个源目标采样 5,000 条，五源共至多 25,000 条；不足时警告并使用全部合格样本，实际数量以 `manifest.json` 为准。默认测试每个 Twwilght 源 500 条、扫描压缩比 `[4,8,16,32]`。NarrativeQA 提供训练与数据适配，测试入口暂不启用。

所有新训练、预处理、benchmark 和 Excel 路径遵守不覆盖已有产物的约束。旧脚本保存在 `legacy/` 用于核对；其路径已归档，不能作为新实验入口直接使用。旧 checkpoint 的转换见 [05 推理指南](05_benchmark.md)。

五数据集下载入口 `scripts/benchmark/sh/10_download_data.sh` 现默认 `DOWNLOAD_MODE=local`：先把固定 commit 的源分片下载到 `artifacts/source_cache/`，再从本地解码并输出 raw JSONL；支持截断响应重试、可用时的分片续传以及分阶段/行数进度。达到训练样本上限后不再下载后续分片。`DOWNLOAD_MODE=stream` 可显式使用旧远端读取路径。

若遇到 `IncompleteRead`、`ChunkedEncodingError` 或解析文件列表后长时间无输出，请先按 [02 下载指南第5节](02_download.md#5-incompleteread--chunkedencodingerror-修复) 更新代码并做20条样本验证。分片缓存应保留，raw JSONL 转换中断则从头转换；两种恢复粒度不同。此下载模式属于五数据集入口，NIAH 独立下载入口的当前行为见 [NIAH 下载指南](../niah/02_download.md)。
