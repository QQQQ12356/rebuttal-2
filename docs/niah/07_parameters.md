# 07 NIAH 脚本变量索引

NIAH Shell 位于 scripts/niah/sh，Python 位于 scripts/niah/py。标量默认值可用环境变量覆盖；数组直接编辑脚本或在命令末尾传同名 Python 参数。优先级：末尾显式 Python 参数 > Shell 传入值 > 对应 JSON 默认。所有相对路径基于项目根，入口会自动 cd。

## 1. 公共与下载

| 脚本 | 变量 | 默认/含义 |
| --- | --- | --- |
| _common.sh | PYTHON_BIN | python；可设环境解释器绝对路径 |
| 同上 | NIAH_ROOT | artifacts/niah；派生 raw/prepared/runs/benchmarks/logs，不改变配置文件位置 |
| 同上 | HF_ENDPOINT | https://huggingface.co；可改镜像，只影响 Hub |
| 同上 | HF_HOME | 项目/artifacts/hf_cache；可与 QA 共享 |
| 同上 | TOKENIZERS_PARALLELISM / OMP_NUM_THREADS | false / 4；控制线程争用 |
| 00_setup_environment.sh | TORCH_INDEX | CUDA12.6 wheel 源 |
| 同上 | INSTALL_DEEPSPEED | 1；0 仅普通 DDP |
| 01_download_model.sh | MODEL_ID | Qwen/Qwen3-4B-Instruct-2507 |
| 同上 | MODEL_PATH | artifacts/models/Qwen3-4B-Instruct-2507，可复用既有模型 |
| 同上 | REVISION | main；推荐固定 commit |
| 10_download_data.sh | REGISTRY | configs/niah/sources.json，角色与 split 在此修改 |
| 同上 | RAW_DIR | NIAH_ROOT/raw |

下载脚本额外接受 `--sources name ...` 与 `--limit N`。--limit=0 是全部；原始 limit 为前缀范围，不是全源随机采样。

## 2. 数据准备

| 脚本 | 变量 | 默认/含义 |
| --- | --- | --- |
| 20_prepare_train.sh | PREPARE_CONFIG | configs/niah/prepare_train.json；配额、长度、候选与训练源 |
| 同上 | MODEL_PATH / RAW_DIR | 默认本地模型 / NIAH_ROOT/raw |
| 同上 | PREPARED_DIR | NIAH_ROOT/prepared/train_v1；必须全新 |
| 同上 | SYNTHETIC_SAMPLES | 空用 JSON；0 禁用合成源 |
| 同上 | SUPERVISED_SAMPLES | 空用每源 JSON；0 使用全部合格监督样本 |
| 21_prepare_grid.sh | GRID_CONFIG | configs/niah/grid.json；可切 grid_128k.json |
| 同上 | MODEL_PATH / RAW_DIR | 与预处理训练保持 tokenizer 一致 |
| 同上 | GRID_DIR | NIAH_ROOT/prepared/grid_v1；与训练目录分开 |
| 同上 | TRAINING_MANIFEST | 空时尝试默认 train_v1 manifest；自定义训练目录应显式填写 |

改训练样本配额、候选池或输入上限：编辑 prepare_train.json。改测试横轴、纵轴、重复数、task：编辑新 grid JSON 并生成新 GRID_DIR。改训练比例模式/数组：编辑 train.json。不要只改文件名而继续读取旧 manifest。

## 3. 训练

| 30_train.sh 变量 | 默认 | 含义 |
| --- | --- | --- |
| CUDA_VISIBLE_DEVICES / NPROC_PER_NODE | 0,1,2,3 / 4 | 每节点物理 GPU 列表 / 进程数 |
| NNODES / NODE_RANK | 1 / 0 | 多节点数量与当前节点编号 |
| MASTER_ADDR / MASTER_PORT | 127.0.0.1 / 29600 | 静态 rendezvous 地址/端口；多节点改为主节点可访问地址 |
| BACKEND | zero2 | ddp/zero2/zero2_cpu/zero3_cpu |
| CONFIG | configs/niah/train.json | 独立训练 JSON |
| MODEL_PATH / PREPARED_DIR | 默认模型 / NIAH_ROOT/prepared/train_v1 | 覆盖 JSON 的模型/数据路径 |
| RUN_NAME | niah_qwen3_4b | 新实验的人类标签 |
| RUN_DIR | 自动 UTC 时间+随机后缀 | 显式新输出目录；多节点所有节点需同一个共享目录 |
| RESUME | 空 | native checkpoint-N；非空时 RUN_DIR 默认取其父目录 |
| MAX_STEPS | 空 | 空用配置；正数是累计 optimizer-step 目标 |
| PLAN_ONLY | 0 | 1 验证并打印计划，无模型/CUDA分配 |

固定/随机比例具体字段、完整 training 参数见 [04](04_training.md)。周期日志写 NIAH_ROOT/logs，避免与 QA 日志混放。模型/数据内容与配置要靠归档哈希追溯，RUN_NAME 仅为标签。

## 4. 合并、测试、打分和作图

| 脚本 | 变量 | 默认/作用 |
| --- | --- | --- |
| 40_merge_lora.sh | CHECKPOINT | 必改为真实 final/checkpoint-N |
| 同上 | MERGED_DIR / MERGE_DEVICE | CHECKPOINT 后接 _merged / cpu |
| 50_benchmark.sh | CUDA_VISIBLE_DEVICES / NPROC_PER_NODE | 0 / 1；每张卡完整双模型 |
| 同上 | CHECKPOINT | 必改为真实 merged 或 adapter 目录 |
| 同上 | GRID_DIR | NIAH_ROOT/prepared/grid_v1 |
| 同上 | OUTPUT_DIR | 自动唯一 NIAH benchmark 目录 |
| 同上 | RATIOS | [4,8,16,32]；Bash 数组，或末尾 --ratios 16 32 |
| 同上 | BATCH_SIZE / WORKERS | 1 / 2 |
| 同上 | WARMUP_BATCHES | 1；不漏评真实样本 |
| 同上 | MAX_NEW_TOKENS / REPETITION_PENALTY | 64 / 1.0；解码预算和重复惩罚 |
| 同上 | PLAN_ONLY | 0；1 检查元数据/网格，无模型分配 |
| 60_score.sh | PREDICTIONS | 必改为真实 predictions.jsonl |
| 70_plot_heatmap.sh | BENCHMARK_DIR | 必改为真实 benchmark 根目录 |
| 同上 | INPUTS | BENCHMARK_DIR/ratio_*/metrics.json，加引号的数组 |
| 同上 | FIGURE_PATH | BENCHMARK_DIR/figures/NIAH.png；另生成同 stem PDF/CSV |
| 同上 | METRIC | accuracy；可选 em/f1/hit_rate |
| 同上 | COLUMNS / DPI | 2 / 300 |
| 同上 | TITLE | SeCo Needle-in-a-Haystack |

90_validate.sh 仅使用公共解释器，运行独立静态检查、Shell 语法和 Linux CPU 用例。它不下载模型/数据，不生成真实 benchmark。
