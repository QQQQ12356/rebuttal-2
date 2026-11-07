# 07 小规模端到端验证

此验证覆盖数据读取、预算筛选、训练、checkpoint 加载、预测、评分和绘图。需要已下载模型和四份数据。以下路径用于首次运行，重复验证时为输出目录设置新版本名。

```bash
export RULER_QA_ROOT=artifacts/ruler_qa
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MODEL_PATH=artifacts/models/Qwen3-4B-Instruct-2507

mkdir -p "$RULER_QA_ROOT"
python - <<'PY'
import json
from pathlib import Path
import os

settings = json.loads(Path("configs/ruler_qa/prepare.json").read_text())
settings.update(train_lengths=[2048], test_lengths=[2048, 4096],
                train_candidates_per_task=128, train_samples_per_task=4,
                test_samples_per_task=2)
path = Path(os.environ["RULER_QA_ROOT"]) / "prepare_smoke.json"
path.write_text(json.dumps(settings, indent=2) + "\n")
PY

export PREPARE_CONFIG="$RULER_QA_ROOT/prepare_smoke.json"
export PREPARED_DIR="$RULER_QA_ROOT/prepared/smoke_train_v1"
export TEST_DIR="$RULER_QA_ROOT/prepared/smoke_test_v1"
export RUN_DIR="$RULER_QA_ROOT/runs/smoke_v1"
export OUTPUT_DIR="$RULER_QA_ROOT/benchmarks/smoke_v1"
export RATIOS='4 8'

bash scripts/ruler_qa/sh/20_prepare_data.sh
SPLIT=test PREPARED_DIR="$TEST_DIR" bash scripts/ruler_qa/sh/20_prepare_data.sh
PLAN_ONLY=1 MAX_STEPS=1 bash scripts/ruler_qa/sh/30_train.sh
MAX_STEPS=1 bash scripts/ruler_qa/sh/30_train.sh

export CHECKPOINT="$RUN_DIR/final"
PLAN_ONLY=1 bash scripts/ruler_qa/sh/50_benchmark.sh
bash scripts/ruler_qa/sh/50_benchmark.sh
PREDICTIONS="$OUTPUT_DIR/ratio_4/predictions.jsonl" bash scripts/ruler_qa/sh/60_score.sh
BENCHMARK_DIR="$OUTPUT_DIR" bash scripts/ruler_qa/sh/70_plot_heatmap.sh
```

- 训练 JSONL 应有 8 行，每任务 4 行。测试 JSONL 应有 8 行，每任务每长度 2 行。
- 评测计划应显示 16 次生成。每个压缩比应有 8 条合并预测、4 个指标格子。
- PNG 和 PDF 应各包含两个任务面板。CSV 应包含 8 条指标记录，不含表头。
- 校验失败时按日志检查数据目录、模型版本或预算。生成显存不足时保留 batch 1，减少网格最长长度后重新预处理。训练显存不足时可降低训练长度、LoRA 秩或使用可用的 ZeRO 后端。
- 通过后用新 shell 或清除本节导出的路径变量，再用正式设置执行 [实验入口](00_README.md)。
