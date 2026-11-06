# 09 验证记录、Linux 检查与排错

## 1. 当前验证范围

本次在 Windows 中仅进行了代码编辑和静态分析，没有安装训练依赖，没有下载模型/数据，没有运行预处理、训练、推理或 Excel 生成。

已执行的静态检查：

| 检查 | 结果/范围 |
| --- | --- |
| Python AST 解析 | 新 seco、scripts/py、tests 目录全部 Python 文件可解析 |
| 原始方法对照 | 两个压缩方法与 legacy 原文件 AST 相同 |
| JSON 解析 | configs 下所有配置可解析 |
| Shell 语法 | 使用 Git Bash 对新 scripts/sh 全部 `.sh` 执行 bash -n 通过；没有执行脚本正文 |
| LF 检查 | 新 Shell 文件未含 CRLF |
| Ruff | 当前 Windows 运行时未安装 Ruff；完整 Ruff 检查留待 Linux 安装工具依赖后执行 |

不能将上述静态通过解释为 GPU 可训练性、ZeRO 恢复正确性、下载成功率或实际加速已经验证。尤其是数据源实时结构、镜像转换分支、显存上限和 CPUAdam 编译，需要在目标 Linux 环境检查。

## 2. Linux CPU 测试

```bash
python -m pip install -r requirements.txt -r requirements-tools.txt
bash scripts/sh/90_validate.sh
```

入口依次执行 Ruff、项目静态检查、Shell 语法以及 `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v`。测试只使用随机小张量、随机初始化的 tiny Qwen3 与临时小文件；不下载 pretrained 模型，不需要 GPU。

| 测试 | 目的 |
| --- | --- |
| test_data.py | Twwilght 干扰文档保留、两种 multi-hop context schema、NQ 多 span/yes-no、Narrative 全文、筛选预算、固定种子采样 |
| test_metrics.py | 归一化、别名、部分重叠、空文本、micro/macro、重复 shard ID 拒绝 |
| test_model.py | 原始/分块压缩输出与梯度、短输入/纯 prompt、真实 EOS mask、变长前缀左 padding、双分支 LoRA 梯度、tiny Qwen 合并前后 greedy 结果 |
| test_reporting.py | 数值/百分比、预测内容作为文本、长正文完整拆分再拼接 |

这些是已编写、待 Linux 执行的测试，未伪造运行结果。如果在 GPU BF16/FP16 对比原始与分块路径，应另记录最大误差、中心/分配一致性与最终生成一致性；近似相似度平局可能对浮点实现敏感。

## 3. 小规模 GPU smoke test

1. 使用新的 RAW_DIR 下载小候选池，四个 benchmark 保持原测试身份。
2. 用新的 PREPARED_DIR，将每源训练/测试样本降为 8–32，输入长度先设 2048，解码预算满足实际样本。
3. 复制 `configs/train.json` 为 `configs/smoke.json`；将 max_steps=2、save_steps=1、logging_steps=1。保留目标 LoRA 设置与真实双模型。
4. 先单卡 `BACKEND=ddp NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 CONFIG=configs/smoke.json ...` 运行两步，确认 loss 有限、checkpoint-1/2 和 final 完整。
5. 在 checkpoint-1 上用同配置恢复到第 2 步，使用原 RUN_DIR；确认 Trainer 状态、global_step 和随机比例 RNG 正常恢复。
6. 在独立新 run 上分别执行两卡 DDP、zero2、zero2_cpu、zero3_cpu；每种检查保存与恢复，不混用 backend。
7. 用少量样本比较 adapter `--no-merge` 与 merged 推理，固定 attention/dtype/比例/解码参数。比较单卡和多卡的 ID 数量与 EM/F1，确认没有漏/重样本。
8. 导出 Excel，抽查一个样本的预测、参考答案、EM/F1 与 JSONL 一致，再扩大样本、长度和步数。

CPU 等价性测试不能代替 ZeRO 的真实集成验证。需要报告实验结果时保留 benchmark.json、manifest、配置、checkpoint 身份、日志与软件版本。

## 4. 常见问题

| 症状 | 检查与处理 |
| --- | --- |
| tokenizer fingerprint mismatch | 预处理与训练/推理模型不一致；检查 MODEL_PATH、tokenizer 文件和版本，重新生成 prepared |
| No eligible narrativeqa examples | 全文太长或候选故事集中；查看 long_input/long_decoder 计数，扩大候选池或预算，禁止静默改用摘要 |
| 请求 5000 但只选到较少条 | 看 eligible/selected、去污染和长度分布；需要严格配额用 STRICT_COUNT=1 |
| No converted parquet files / revision not found | 核对 MultiNews 转换分支和 endpoint；切官方源或使用可追溯本地原始 JSONL |
| Frozen base weight differs | 旧 checkpoint 导入用了错误基座/dtype，或基座确实改过；不能丢弃差异继续导入 |
| Adapter keys mismatch | 模型/LoRA rank/文件格式不一致；使用正确 experiment.json，旧文件走 41 转换 |
| CUDA OOM，ZeRO-2 也无改善 | 双基座仍每卡复制；降低输入/批大小、减 similarity_chunk_tokens 或使用 zero3_cpu |
| 聚类耗时较高 | 原始算法包含逐簇 Python 循环；本次保持原方法，不保证聚类部分加速；用一致条件实测 |
| CPUAdam 编译失败 | 检查 gcc/g++、ninja、CUDA toolkit 与 torch 匹配；先验证 ddp/zero2，再修复卸载环境 |
| bf16 不支持 | 换支持 BF16 的 GPU，或在新配置中用 float16；保持对照精度一致 |
| Resume target already reached | MAX_STEPS 必须大于断点 global_step；或按 epoch 目标确认训练剩余量 |
| Resume config/backend/world size mismatch | 恢复须同配置/同进程数，只允许更改累计步数；其他改变开新 run |
| rank hang / NCCL error | 检查 GPU 数、进程数、端口、各 rank 首个报错、多节点网络；不要只看最后的 barrier 超时 |
| run/benchmark/prepared 目录已存在 | 新实验换目录；恢复训练用 RESUME；不自动覆盖已完成产物 |
| 指标整体为零 | 先检查 prediction 与 answer、模型权重身份、生成长度、输入 prompt；不要先修改 metric 归一化掩盖问题 |
| Excel 没有正文 | benchmark 默认不保存 input；启用 INCLUDE_CONTEXT=1 重新生成预测，已有文件不能凭空还原 |

如需单独静态检查可用 `python scripts/py/90_validate.py`；它不加载项目 ML 模块，也不训练。完整实验命令的参数列表可用对应 Python 文件 `--help` 查看。
