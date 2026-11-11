# 01 Linux 环境和模型下载

在项目根目录创建并激活 Python 3.10 或 3.11 环境。默认安装 PyTorch 2.8 CUDA 12.6 构建，Python 依赖版本见根目录 `requirements*.txt`。安装与实验均在 Linux 进行。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
method=comi                         # seco / comi / icae / ram
for task in benchmark niah ruler_qa; do
    INSTALL_DEEPSPEED=0 bash "scripts/$task/sh/$method/00_setup_environment.sh"
done
```

上例安装三个任务的依赖并使用 DDP。需要 ZeRO 时，将安装命令的 `INSTALL_DEEPSPEED` 设为 1。只做单个任务时只需执行对应的安装入口。CUDA 构建由 `TORCH_INDEX` 指定，应与 Linux GPU 驱动兼容。用 `nvidia-smi` 检查可见 GPU。

在 Hugging Face 模型页完成访问授权，并在 Linux 环境用 `hf auth login` 登录，或提供有读取权限的 `HF_TOKEN`。[Llama 3.2 1B Instruct 模型页](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)。

```bash
bash "scripts/benchmark/sh/$method/01_download_model.sh"
# 固定 Hub 版本或指向独立快照目录
REVISION=your_commit_sha MODEL_PATH=/models/llama32_1b bash "scripts/benchmark/sh/$method/01_download_model.sh"
```

每个方法只需下载一次，三条任务流程默认共用同一方法目录下的 `model/`。也可为四种方法设置同一个只读 `MODEL_PATH`。下载脚本记录解析出的 Hub commit。替换基座时，同时设置 `BACKBONE` 目录名、`MODEL_ID` 或 `MODEL_PATH`，重新预处理并重新训练。 `BACKBONE` 控制路径命名，实际权重由 `MODEL_PATH` 或训练配置决定。
