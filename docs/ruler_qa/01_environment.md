# 01 Linux 环境

需要 Bash、curl、Python 3.10 或 3.11。下载、预处理、离线评分和绘图可在 Linux CPU 主机完成。训练和生成评测需要可见的 NVIDIA CUDA GPU。默认 BF16，需 GPU 和驱动支持。模型同时加载编码器和解码器，评测每个 GPU 各加载一对。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
curl --version
nvidia-smi
bash scripts/ruler_qa/sh/00_setup_environment.sh
```

| 设置 | 默认 | 如何调整 |
| --- | --- | --- |
| `PYTHON_BIN` | `python` | 指向当前 Linux venv 的 Python，建议用绝对路径 |
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cu126` | 按 Linux 驱动支持的 CUDA wheel 索引修改 |
| `INSTALL_DEEPSPEED` | `0` | 设 `1` 安装 ZeRO 后端依赖。CPU offload 还需要主机内存及可用的编译工具 |
| `HF_HOME` | 项目下 `artifacts/hf_cache` | 指向共享缓存或空间充足的盘 |
| `HF_ENDPOINT` | `https://huggingface.co` | 指定模型下载所用的 Hub 地址。数据下载 URL 由 `sources.json` 管理 |

已有 SeCo Linux 环境时，在激活后执行 `python -m pip install -r requirements-ruler-qa.txt` 和 `python -m pip check`，保留已匹配驱动的 PyTorch CUDA wheel。可用以下命令检查运行环境。

```bash
python -c 'import torch, transformers, ijson, matplotlib; print(torch.__version__, torch.version.cuda, transformers.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())'
python -c 'import torch; print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "no CUDA GPU")'
```

长上下文成本随 token 数增长。先以单卡、batch 1 和短长度完成 [小规模验证](07_smoke_test.md)，再增加长度和并行度。原始 JSON 约 0.66–0.68 GB，另外为模型、缓存、预处理 JSONL、训练检查点和可选的双模型合并目录预留磁盘。
