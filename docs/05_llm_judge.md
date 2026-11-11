# 05 NIAH 和 RULER QA 的 LLM-as-Judge

Judge 使用已有的 `predictions.jsonl`。它逐条提交问题、参考答案、候选答案，按 1–10 分评分。量表锚点为 1 完全错误或空答，3 轻微相关，5 部分正确，7 基本正确但有遗漏，10 完全正确。NIAH 按序列长度和深度聚合，RULER QA 按任务和目标长度聚合。量表和汇总方式参考 Activation Beacon 的 NIAH 评估。[Activation Beacon 评估实现](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/Long_LLM/activation_beacon/main/eval_needle.py)。

```bash
method=comi
task=niah                         # niah / ruler_qa
export JUDGE_MODEL=your-chat-model
# 外部服务凭据由 JUDGE_API_KEY 或 OPENAI_API_KEY 提供
PREDICTIONS=/path/to/eval/ratio_4/predictions.jsonl bash "scripts/$task/sh/$method/65_judge.sh"

# 使用本地提供 /v1/chat/completions 的服务
JUDGE_BASE_URL=http://127.0.0.1:8000/v1 JUDGE_MODEL=my-judge \
    PREDICTIONS=/path/to/eval/ratio_4/predictions.jsonl bash "scripts/$task/sh/$method/65_judge.sh"
```

| Bash 变量 | Python 参数 | 作用 |
| --- | --- | --- |
| `PREDICTIONS` | `--predictions` | 单个压缩比的预测文件 |
| `JUDGE_MODEL` | `--judge-model` | Judge 服务识别的聊天模型 ID |
| `JUDGE_BASE_URL` | `--base-url` | 兼容 Chat Completions 的 API 根地址，默认 `https://api.openai.com/v1` |
| `JUDGE_TIMEOUT` | `--timeout` | 单次 HTTP 请求超时，默认 60 秒 |
| `JUDGE_ATTEMPTS` | `--attempts` | 每样本最多请求次数，默认 4 |
| `JUDGE_API_KEY`、`OPENAI_API_KEY` | 从环境读取 | Bearer 认证，优先使用 `JUDGE_API_KEY` |
| CLI 追加参数 | `--output-dir` | 默认 `ratio_<R>/judge/`。用不同目录保存不同 Judge 模型或口径的结果 |

每条评分写入缓存，重新运行同一命令会复用已经完成的评分。缓存身份包括预测文件哈希、量表哈希、Judge 模型和服务地址。评分结束后写出 `judge_scores.jsonl` 和 `judge_metrics.json`，后者记录整体均分、逐格均分和样本数。原确定性指标仍可通过 `60_score.sh` 重算。报告 Judge 均分时标明 1–10 量表及服务模型版本。
