# 01 环境与依赖

运行目标为 Linux x86_64、Python 3.10/3.11、NVIDIA GPU。预处理和 Excel 导出可用 Linux CPU。当前 Windows 只编辑代码；编号实验命令会拒绝在 Windows 启动，静态检查器例外。

## 1. 创建环境

```bash
conda create -n seco python=3.10 -y
conda activate seco
cd /你的/Linux/路径/seco-rebuttal-2
bash scripts/benchmark/sh/00_setup_environment.sh
```

没有 conda 时使用 `python3.10 -m venv .venv`，然后 `source .venv/bin/activate`。Shell 的 `PYTHON_BIN` 默认是当前环境的 `python`；也可设置其绝对路径。

安装入口默认使用 PyTorch 2.8.0 的 CUDA 12.6 wheel。先用 `nvidia-smi` 检查驱动；驱动、GPU 和工具链不匹配时按 PyTorch 官方安装说明选择同版本的其他 CUDA wheel 源。编译 DeepSpeed CPUAdam/FlashAttention 需要对应的 C/C++ 编译器和 CUDA toolkit，wheel 自带的 CUDA 运行库不等于完整 toolkit。

```bash
# 普通 DDP 不需要 DeepSpeed。
INSTALL_DEEPSPEED=0 bash scripts/benchmark/sh/00_setup_environment.sh

# 显式选择当前环境 Python。
PYTHON_BIN=/opt/conda/envs/seco/bin/python bash scripts/benchmark/sh/00_setup_environment.sh
python scripts/benchmark/py/00_check_environment.py
```

## 2. 依赖分组

| 文件 | 用途 | 关键版本 |
| --- | --- | --- |
| `requirements.txt` | 模型、LoRA、Trainer、数据读取 | torch 2.8.0、transformers 4.57.6、peft 0.17.1、accelerate 1.10.1、datasets 3.6.0 |
| `requirements-deepspeed.txt` | ZeRO 与 CPU 卸载，包含基础依赖 | deepspeed 0.17.5、ninja 1.13.0 |
| `requirements-tools.txt` | Excel 与静态规范检查 | openpyxl 3.1.5、ruff 0.14.3 |

这些是本次编写的复现实验环境约束，尚未在目标 Linux GPU 上联合安装和实测；首先执行 `python -m pip check` 与 [09 验证流程](09_validation.md)。不要在已有训练环境中直接盲目升级依赖。原始 `icecream`、`wandb` 和无实际执行逻辑的 compile 开关不是新流程依赖。

## 3. 网络、镜像与缓存

```bash
export HF_ENDPOINT=https://huggingface.co
# 需要镜像时，在启动 Python 前替换。
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/你的/大容量磁盘/hf_cache
bash scripts/benchmark/sh/01_download_model.sh
bash scripts/benchmark/sh/10_download_data.sh
```

镜像站是否同步目标仓库和转换分支由镜像服务决定。下载失败应先核对 endpoint、revision、权限和网络；程序不会替换成其他数据集。私有/受限仓库可先 `hf auth login`；不要把 token 写入脚本或文档。

五数据集的新版下载入口默认使用本地分片模式，其 `SOURCE_CACHE` 与 `HF_HOME` 是不同位置：HF_HOME 仍供认证、模型和其他 Hub/datasets 调用使用；SOURCE_CACHE 显式保存该入口的源分片，默认 artifacts/source_cache，不会因为只修改 HF_HOME 就迁移。磁盘紧张时把两者分别设到合适位置，并保持 SOURCE_CACHE 以复用完整/未完成分片。

```bash
SOURCE_CACHE=/大容量磁盘/seco_source_cache \
HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60 \
bash scripts/benchmark/sh/10_download_data.sh
```

文件请求超时由 HF_HUB_DOWNLOAD_TIMEOUT 控制；HEAD/ETag 超时由 HF_HUB_ETAG_TIMEOUT 控制；dataset-info API 超时由该脚本的 METADATA_TIMEOUT 单独控制，默认60秒。它们不是整次命令的总时限，SDK内部也可能重试。该 Shell 入口默认 HF_HUB_DISABLE_XET=1、HF_HUB_ENABLE_HF_TRANSFER=0，选择 HTTP 部分文件恢复路径；直接调用 Python 时如需相同行为，应在启动 Python 前显式设置这些环境变量。更多故障处理见 [下载指南](02_download.md#5-incompleteread--chunkedencodingerror-修复)。

完整下载并预处理后可以启用离线模式：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
MODEL_PATH=/本地/Qwen3-4B-Instruct-2507 bash scripts/benchmark/sh/30_train.sh
```

离线前必须确保模型文件、tokenizer、raw/prepared 产物均已就绪。合并后的模型包含 encoder、decoder、tokenizer，推理不再需要原始基座下载。

## 4. 显存与主机内存

SeCo 保留两份独立 4B 基座。BF16 基座权重本身约 16 GB 十进制总量，另有 LoRA、激活、输出 logits、聚类相似度、优化器与 KV cache。不能用单个 4B 模型的显存需求估计 SeCo。

ZeRO-2 分片优化器/梯度，仍在每张 GPU 保留全部基座参数；ZeRO-2 CPU 卸载也不会卸载基座权重。若权重与激活仍放不下，使用 ZeRO-3 CPU 卸载，减少输入长度或 batch。LoRA 优化器占比相对小，ZeRO-2 optimizer offload 的收益可能有限。CPU 卸载和 CPU 合并都需要足够 RAM，可能显著降低速度；实际可用长度需要目标硬件 smoke test。

默认 `sdpa` 不要求安装 `flash-attn`。使用 `flash_attention_2` 时自行安装与 torch/CUDA/GPU 一致的版本，然后先运行小样本验证；安装脚本不会自动编译 FlashAttention。
