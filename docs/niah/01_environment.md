# 01 NIAH 环境

运行要求与原 SeCo 一致：Linux、Python 3.10/3.11、CUDA GPU。数据构建、指标重算、热力图与 NIAH CPU 测试可用 Linux CPU；真实模型生成使用 GPU。当前 Windows 不执行这些实验。

[训练集评测](09_train_benchmark.md) 与留出评测共用推理环境。训练集预检只需要 Python 和本地预处理产物及 checkpoint 元数据，正式运行使用 CUDA。

已有本工程环境时只需要补齐绘图依赖：

```bash
conda activate seco
cd /你的路径/seco-rebuttal-2
python -m pip install -r requirements-niah.txt -r requirements-tools.txt
python -m pip check
```

新环境可按独立入口安装：

```bash
conda create -n seco_niah python=3.10 -y
conda activate seco_niah
bash scripts/niah/sh/00_setup_environment.sh
```

| 文件/变量 | 默认 | 用途 |
| --- | --- | --- |
| requirements-niah.txt | 基础 requirements + matplotlib 3.10.7 | 训练/分词依赖和静态科学绘图；不需要 seaborn/pandas/外部评委 |
| requirements-tools.txt | openpyxl、ruff | 静态检查；openpyxl 保持工程工具一致，不是 NIAH 绘图前提 |
| requirements-deepspeed.txt | 基础依赖 + DeepSpeed | 选 zero2/zero2_cpu/zero3_cpu 时安装 |
| PYTHON_BIN | python | 当前环境解释器，可用绝对路径 |
| TORCH_INDEX | CUDA 12.6 torch wheel 源 | 与 Linux 驱动/toolkit 一起核对 |
| INSTALL_DEEPSPEED | 1 | 0 仅安装普通 DDP 依赖 |

没有 conda 时使用 Python venv。CUDA/DeepSpeed CPUAdam 的安装提示见 [Benchmark 环境指南](../benchmark/01_environment.md)。新增环境约束尚未在目标 Linux GPU 上联合安装实测，先做 [08 验证](08_validation.md)。

热力图采用 Matplotlib Agg 后端，可在无桌面、无显示器的服务器执行，默认英文字体与标签避免 CJK 字体缺失。没有 OpenAI API key 或其他在线 LLM 评委依赖。

```bash
export HF_ENDPOINT=https://huggingface.co
# Hub 来源可切镜像；Paul Graham 的 GitHub 下载不受 HF_ENDPOINT 控制。
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/大容量磁盘/hf_cache
```

完成模型和数据下载后可设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`，后续使用本地模型/raw/prepared。完整 merged 模型包含 tokenizer，不再依赖基座在线访问。

SeCo 仍有两份独立 4B 基座，BF16 权重约 16 GB 十进制总量；额外还需长输入激活、聚类张量、生成 cache。**128K 是可选实验网格，不是显存可运行承诺**。默认 SDPA、分块余弦不会消除 encoder 长注意力成本；先做 2K/4K 与 batch=1 验证，再依据实际显存扩大长度。ZeRO-3 可帮助训练参数分片，但当前测试数据并行仍要求每张卡放下完整双模型。
