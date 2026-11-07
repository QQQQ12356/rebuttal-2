# SeCo 实验指南

Benchmark 和 NIAH 共用 SeCo 模型、语义压缩算法及训练与合并实现。默认基座是 `Qwen/Qwen3-4B-Instruct-2507`。在 Linux 上执行实验命令，Windows 用于编辑和静态检查。

| 流程 | 实验指南 | 脚本 | 配置 |
| --- | --- | --- | --- |
| Benchmark | [五数据集训练、四数据集评测与 Excel 导出](docs/benchmark/00_README.md) | `scripts/benchmark/{py,sh}/` | `configs/benchmark/` |
| NIAH | [大海捞针训练、网格评测与热力图](docs/niah/00_README.md) | `scripts/niah/{py,sh}/` | `configs/niah/` |

共享实现位于 `seco/`，共享 ZeRO 配置位于 `configs/zero*.json`。实验用例分别位于 `tests/benchmark/` 和 `tests/niah/`，共享模型验证和原始压缩基线位于 `tests/`。原始 SeCo 脚本保存在 `legacy/` 供核对。

在项目根目录执行各指南中的命令。已有 Benchmark 命令中的 `scripts/py/` 和 `scripts/sh/` 分别改用 `scripts/benchmark/py/` 和 `scripts/benchmark/sh/`。已生成的数据、检查点、评测结果和日志仍使用原来的 `artifacts/` 与 `logs/` 路径。
