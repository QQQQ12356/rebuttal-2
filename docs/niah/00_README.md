# 00 独立 NIAH 实验入口

本流程提供大海捞针训练、长度×位置测试和热力图，不调用之前五数据集的下载、预处理或 benchmark 脚本。它复用 SeCo 模型、原始语义压缩算法、Trainer 和 LoRA 合并代码；未修改核心方法。默认模型是 `Qwen/Qwen3-4B-Instruct-2507`，压缩比保留随机/固定模式。

**Windows 只修改代码和做静态检查；以下下载、预处理、训练、推理、绘图及单元测试都在 Linux 执行。当前没有生成或声称获得真实 NIAH 准确度图。**

| 编号 | 指南 | 重点 |
| --- | --- | --- |
| 01 | [环境](01_environment.md) | 复用环境、NIAH 绘图依赖、Linux 安装 |
| 02 | [数据下载与来源](02_download.md) | 两个有 train split 的检索数据集、PG19、Paul Graham、镜像 |
| 03 | [训练数据预处理](03_prepare_train.md) | 配额、候选池、长度过滤、重新分词、随机针 |
| 04 | [训练与恢复](04_training.md) | 独立配置、随机/固定压缩比、多卡与 CPU 卸载 |
| 05 | [测试网格生成](05_prepare_grid.md) | 精确序列长度、位置定义、paired trials、128K 配置 |
| 06 | [测试、指标与绘图](06_benchmark_plot.md) | EM/F1、分片合并、单图多比例、PNG/PDF/CSV |
| 07 | [脚本变量索引](07_parameters.md) | 每个 Shell 变量、参数优先级、修改位置 |
| 08 | [验证与排错](08_validation.md) | 已做静态检查、待做 Linux 测试、smoke test |
| 09 | [训练集评测](09_train_benchmark.md) | 显式开启、校验原训练数据、按来源与任务汇总 |

```text
configs/niah/           sources.json、prepare_train.json、train.json、grid*.json
scripts/niah/py/        独立 Python 命令
scripts/niah/sh/        独立 Bash 入口与变量注释
seco/niah/             针、背景窗口、网格指标、热力图
tests/niah/            独立 Linux CPU 验证用例
artifacts/niah/         raw、prepared、runs、benchmarks、logs
```

`seco/train_cli.py` 和 `seco/merge_cli.py` 集中共享训练/合并实现；原 QA 入口保持原有默认配置，NIAH 入口指定独立默认配置并验证数据 task，防止误把两种 prepared 数据混用。模型下载目录和 HF cache 可复用以节省磁盘；实验数据、模型 run 与测试结果各自分开。

Linux 上先激活环境并进入项目根，然后依次执行：

```bash
bash scripts/niah/sh/00_setup_environment.sh
# 若原默认模型已完整下载，可以跳过下一行。
bash scripts/niah/sh/01_download_model.sh
bash scripts/niah/sh/10_download_data.sh
bash scripts/niah/sh/20_prepare_train.sh
bash scripts/niah/sh/21_prepare_grid.sh
PLAN_ONLY=1 bash scripts/niah/sh/30_train.sh
bash scripts/niah/sh/30_train.sh
```

训练日志打印实际 run 路径。填写真实路径后合并、测试和绘图：

```bash
export CHECKPOINT=artifacts/niah/runs/你的实际run/final
bash scripts/niah/sh/40_merge_lora.sh
CHECKPOINT="${CHECKPOINT}_merged" bash scripts/niah/sh/50_benchmark.sh

# 替换为测试日志打印的真实目录。
BENCHMARK_DIR=artifacts/niah/benchmarks/你的实际benchmark \
    bash scripts/niah/sh/70_plot_heatmap.sh
```

默认训练目标：4,000 条 PG19 合成针 + 各 2,000 条 CogitoProbe-Bits/Bind，实际以筛选统计为准。默认测试为 4 个长度×11 个深度×5 个 trial=220 条，每个压缩比各测完整同一网格。默认四个比例共 880 个计分样本，另有不计分的预热生成。

如果要诊断训练集表现，用训练时的 `PREPARED_DIR` 和对应 checkpoint 运行 `EVAL_TRAIN=1 bash scripts/niah/sh/51_benchmark_train.sh`。参数、预检和产物见 [09 训练集评测](09_train_benchmark.md)。

最终 `figures/NIAH.png` 可在一张图中展示多个比例的热力图，另存矢量 PDF 和逐格 CSV。布局参考用户 `NIAH.png`；默认颜色表示 0–100% 的检索 EM 成功率，区别于参考图中的 GPT 1–10 主观评分。
