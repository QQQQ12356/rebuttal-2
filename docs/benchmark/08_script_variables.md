# 08 Shell 变量与修改位置索引

每个编号 `.sh` 顶部有默认值和注释。标量采用 `${变量:-默认值}`，可直接编辑文件，也可用 `变量=值 bash 脚本` 临时覆盖。数组 `DATASETS/TRAIN_SAMPLES/RATIOS/INPUTS` 直接编辑脚本，或在末尾追加相应 Python CLI 参数覆盖。命令行重复标量参数时 argparse 使用最后一个值；数组参数也由最后一次赋值替换。

所有入口切换到项目根后执行，路径中的空格通过引号保留。相对产物路径基于项目根。`--config` 文件内部的 prepared 路径同样基于项目根，不基于配置文件目录。

## 1. 公共变量 `_common.sh`

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| PYTHON_BIN | python | 激活环境的 Python；允许绝对路径；不应写成带额外参数的整条命令 |
| HF_ENDPOINT | https://huggingface.co | 运行前可换镜像 endpoint |
| HF_HOME | 项目/artifacts/hf_cache | Hub/datasets 公共缓存，建议放大容量磁盘 |
| TOKENIZERS_PARALLELISM | false | 固定关闭 tokenizer 线程嵌套 |
| OMP_NUM_THREADS | 4 | 每进程 OpenMP 线程上限，CPU 争用时减小 |

`HF_HUB_OFFLINE/HF_DATASETS_OFFLINE` 可由外部环境设置；下载时应关闭，完整本地训练时可设 1。`SCRIPT_DIR/PROJECT_ROOT` 是自动计算的内部路径，不需改动。

## 2. 环境与下载

| 脚本 | 变量 | 默认/用途 |
| --- | --- | --- |
| 00_setup_environment.sh | TORCH_INDEX | CUDA 12.6 torch wheel 源，按驱动适配 |
| 同上 | INSTALL_DEEPSPEED | 1；0 跳过 ZeRO 依赖 |
| 01_download_model.sh | MODEL_ID | meta-llama/Llama-3.2-1B-Instruct |
| 同上 | REVISION | main；最终复现建议 commit |
| 同上 | MODEL_PATH | artifacts/seco/Llama-3.2-1B-Instruct/model |
| 10_download_data.sh | REGISTRY | configs/benchmark/datasets.json；修改各源 repo/config/split/revision |
| 同上 | RAW_DIR | artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw |
| 同上 | ROLE | all；可改 training 或 benchmark |
| 同上 | TRAIN_LIMIT | 30000；每训练源原始前 N 行，0 全部 |
| 同上 | DOWNLOAD_MODE | local；先下载分片到本地再解码，stream 为旧远端模式 |
| 同上 | SOURCE_CACHE | artifacts/source_cache；保留完整文件和 .incomplete 以供续传 |
| 同上 | DOWNLOAD_ATTEMPTS / RETRY_DELAY | 6 / 2；失败重试与初始退避，最大30秒 |
| 同上 | METADATA_TIMEOUT | 60秒；dataset-info API 超时，独立于文件与ETag超时 |
| 同上 | ROW_BATCH_SIZE / PROGRESS_EVERY | 16 / 100；本地 Arrow 批大小与输出进度间隔 |
| 同上 | HF_HUB_DOWNLOAD_TIMEOUT / HF_HUB_ETAG_TIMEOUT | 120 / 60秒；用户环境中显式设置的值优先 |
| 同上 | HF_HUB_DISABLE_XET / HF_HUB_ENABLE_HF_TRANSFER | 1 / 0；默认 HTTP 可续传路径，均可显式覆盖 |
| 同上 | PYTHONUNBUFFERED | 固定设为1，使已有日志及时输出；脚本还显式 flush 阶段/行数日志 |

本节新增的 DOWNLOAD_MODE、SOURCE_CACHE、重试及传输默认值属于五数据集的 10_download_data.sh，不是所有入口的公共设置。SOURCE_CACHE 显式指定源分片缓存，不随 HF_HOME 自动迁移；直接执行 Python 时不会自动获得 Shell 中设置的 HF_HUB_* 默认值。详情及服务器更新/小样本命令见 [02 下载指南](02_download.md#5-incompleteread--chunkedencodingerror-修复)。

## 3. 预处理 `20_prepare_data.sh`

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| MODEL_PATH | 默认本地模型目录 | tokenizer 来源，必须与后续训练/测试一致 |
| RAW_DIR / PREPARED_DIR | artifacts/seco/Llama-3.2-1B-Instruct/benchmark/raw / artifacts/seco/Llama-3.2-1B-Instruct/benchmark/prepared/default | 输入根目录 / 全新输出版本 |
| DATASETS | 五个训练源数组 | 与 TRAIN_SAMPLES 按位置对应 |
| TRAIN_SAMPLES | 各 5000 | 合格后抽样目标；Python 也可传单个数 |
| TEST_SAMPLES | 500 | 每个 Twwilght 测试源抽样数 |
| CANDIDATE_LIMIT | 0 | 扫描每训练文件前 N 行；0 全部已下载候选 |
| MAX_INPUT_TOKENS | 16384 | context+prompt 上限 |
| MAX_ANSWER_TOKENS | 1024 | 训练 answer+EOS 上限 |
| MAX_DECODER_TOKENS | 8192 | 压缩前缀+prompt+target/生成预算上限 |
| MIN_CONTEXT_TOKENS | 32 | 正文最短长度 |
| MIN_RATIO | 4 | 训练与测试计划使用的最小压缩比 |
| TEST_GENERATION_BUDGET | 512 | 必须覆盖 QA/NEWS 最大生成 token 数 |
| SEED | 42 | 数据采样种子 |
| STRICT_COUNT | 0 | 1：不足目标数量报错；0：警告并记录实际数 |

## 4. 训练 `30_train.sh`

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| CUDA_VISIBLE_DEVICES | 0,1,2,3 | 本节点物理 GPU 列表 |
| NPROC_PER_NODE | 4 | 本节点进程数，应与参与 GPU 数匹配 |
| NNODES / NODE_RANK | 1 / 0 | 节点总数 / 当前节点编号 |
| MASTER_ADDR / MASTER_PORT | 127.0.0.1 / 29500–29503 | 按 SeCo、COMI、ICAE、RAM 分配，支持覆盖。多节点需可访问主节点 |
| BACKEND | zero2 | ddp / zero2 / zero2_cpu / zero3_cpu |
| CONFIG | configs/benchmark/seco/train.json | LoRA、dtype、attention、压缩比数组、优化器等 |
| MODEL_PATH | 默认本地模型目录 | 覆盖 JSON 基座路径 |
| PREPARED_DIR | artifacts/seco/Llama-3.2-1B-Instruct/benchmark/prepared/default | 覆盖 JSON 数据路径 |
| RUN_NAME | benchmark | 新实验目录的人类标签，实际还追加 UTC 时间和随机后缀 |
| RUN_DIR | 自动唯一目录 | 显式指定实验目录；多节点必须所有节点一致 |
| RESUME | 空 | checkpoint-N 路径；非空时 RUN_DIR 默认取其父目录 |
| MAX_STEPS | 空 | 空使用 JSON；正数覆盖累计 optimizer-step 目标 |
| PLAN_ONLY | 0 | 1 静态计划，无模型分配和训练目录写入 |

## 5. 合并与旧权重迁移

| 脚本 | 变量 | 默认/说明 |
| --- | --- | --- |
| 40_merge_lora.sh | CHECKPOINT | 必改为真实 final/checkpoint-N 目录 |
| 同上 | MERGED_DIR | CHECKPOINT 后接 _merged；必须尚不存在 |
| 同上 | MERGE_DEVICE | cpu；也可 cuda，注意双模型显存 |
| 41_import_checkpoint.sh | SOURCE | 必改为旧未合并双 LoRA safetensors 文件 |
| 同上 | CONFIG | configs/benchmark/seco/train.json；旧模型必须另配对应基座与 r/alpha/dropout |
| 同上 | OUTPUT_DIR | artifacts/seco/Llama-3.2-1B-Instruct/benchmark/imported/original_seco，新格式输出目录 |

基座文件迁移时 Python 提供 `--model /新基座路径`；必须保持相同权重。迁移工具不猜测旧 checkpoint 的 LoRA alpha 或来源数据。

## 6. benchmark、打分与 Excel

| 脚本 | 变量 | 默认/用途 |
| --- | --- | --- |
| 50_benchmark.sh | CUDA_VISIBLE_DEVICES / NPROC_PER_NODE | 0 / 1；每卡完整双模型，数据分片 |
| 同上 | CHECKPOINT | 必改为真实 adapter/merged 目录 |
| 同上 | PREPARED_DIR | artifacts/seco/Llama-3.2-1B-Instruct/benchmark/prepared/default |
| 同上 | OUTPUT_DIR | 自动唯一 benchmark 目录 |
| 同上 | DATASETS / RATIOS | 空使用当前 split 的全部数据集，比例默认 [4,8,16,32] |
| 同上 | BATCH_SIZE / SAMPLES | 1 / 0；SAMPLES 限制每个数据集，0 使用全部 |
| 同上 | EVAL_TRAIN / MAX_TEST_SAMPLES | 默认 0 / 0。前者选训练集，后者限制每个比例的总样本数 |
| 同上 | SHUFFLE / PLAN_ONLY | 默认 0 / 0。可开启种子选样或只查看评测计划 |
| 同上 | WORKERS / WARMUP_BATCHES | 2 / 1；预热不会漏评样本 |
| 同上 | QA_MAX_NEW_TOKENS / NEWS_MAX_NEW_TOKENS | 64 / 512，均不得超过预处理生成预算 |
| 同上 | REPETITION_PENALTY / SEED | 1.0 / 42 |
| 同上 | INCLUDE_CONTEXT | 0；1 保存完整正文供 Excel 导出 |
| 同上 | RECORD_GROUP_STATS | SECO 默认 1；0 关闭逐样本分组统计 |
| 55_export_group_stats.sh | BENCHMARK_DIR / GROUP_STATS_OUTPUT | 已完成的评测目录 / 新 JSONL 路径，默认写入评测目录。见 [分组统计](10_group_stats.md) |
| 60_score.sh | PREDICTIONS | 必改为真实 predictions.jsonl 路径 |
| 70_export_excel.sh | INPUTS | 指标文件/glob 数组，保持引号 |
| 同上 | EXCEL_PATH | 自动唯一 xlsx 路径 |
| 同上 | INCLUDE_DETAILS | 1；0 只导出指标汇总 |

`90_validate.sh` 使用公共 PYTHON_BIN，没有实验超参数。训练和 benchmark 的 LOG_FILE 自动存放任务目录下的 `logs/`；各次日志独立。目录自动命名用 UTC 时间，避免机器时区影响，实验复现实质由保存的配置、种子与哈希决定。

常见修改定位：改每数据集训练数量→20 的 TRAIN_SAMPLES；改随机比例→train.json 的 model；改 ZeRO/卸载→30 的 BACKEND；改 GPU→对应入口 CUDA_VISIBLE_DEVICES/NPROC_PER_NODE；改生成长度→50 的 QA/NEWS 变量，同时检查预处理预算；改测试数量→优先改 20 的 TEST_SAMPLES 重新生成固定测试版本，临时子样本可用 50 的 SAMPLES。
