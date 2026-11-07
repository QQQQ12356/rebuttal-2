# 06 benchmark 结果导入 Excel

Excel 导出在 Linux 执行，不需要桌面 Excel，也不加载模型。安装 `requirements-tools.txt` 后运行：

```bash
bash scripts/benchmark/sh/70_export_excel.sh
EXCEL_PATH=artifacts/reports/paper_table.xlsx bash scripts/benchmark/sh/70_export_excel.sh

python scripts/benchmark/py/70_export_excel.py \
    --inputs 'artifacts/benchmarks/**/metrics.json' \
    --output artifacts/reports/all_runs.xlsx --include-details
```

| 参数/变量 | 默认 | 作用 |
| --- | --- | --- |
| `--inputs` / Shell INPUTS | artifacts/benchmarks/**/metrics.json | 一个或多个指标文件/递归 glob；必须加引号交给 Python 展开 |
| `--output` / EXCEL_PATH | Shell 自动生成唯一文件名 | 新 xlsx 路径，已有文件拒绝覆盖 |
| `--include-details` / INCLUDE_DETAILS | Python 关闭，Shell 为 1 | 从每个指标文件旁的 predictions.jsonl 读取原始预测内容 |

多个 pattern 会先去重文件路径，再合并导出；无匹配项或指标格式错误会直接报错。

| 工作表 | 内容 |
| --- | --- |
| Summary_1 | 每实验/压缩比/数据集的 count、EM、F1、计时、峰值显存、checkpoint、基座、训练比例模式、配置哈希、来源路径；另含 micro/macro 行 |
| Predictions_1 | 每样本 ID、数据集、比例、模型输出、参考答案、重算 EM/F1、prompt、可选完整 context |
| LongText_1 | 超过 30,000 字符的正文/输出/答案按 source、id、field、part 拆分存储，防止 Excel 单格长度限制截断 |
| Readme_1 | 指标尺度、平均规则、计时与长文本说明 |

达到 Excel 单 sheet 行数上限时自动新增同类编号 sheet。长文本按 part 升序拼回；XML 不允许的控制字符编码为字面 `\uXXXX`。包括 `=...` 在内的模型输出都作为普通文本单元格，不执行为公式。EM/F1 保留数值 0–1 并显示百分比，不转为不可计算的字符串。

若需要完整输入内容，benchmark 时先设置 `INCLUDE_CONTEXT=1`。没有保存 context 的结果不能在 Excel 导出阶段凭空恢复正文；仍可导出预测、答案和 prompt。含详情导出可能较大，`INCLUDE_DETAILS=0` 只汇总指标。

新格式识别 `datasets.<name>.count/em/f1` 和可选 `micro/macro`；兼容旧 SeCo 指标的 `subset_summary.<name>.count/avg_em/avg_f1`。指标必须是 0–1，传入 0–100 的旧文件会报错，请先明确转换尺度。旧文件中的平均行缺失时不编造；建议先用 `60_score.py` 从原始 prediction JSONL 重新打分，再导出。

外部结果请将已评分文件命名为 `metrics.json`，相邻预测命名为 `predictions.jsonl`，或用 `--inputs` 指定其他 JSON 指标路径。原始 prediction 文件可使用 `model_output/ground_truth/subset`，详情导出会适配这些旧字段。任意未知 benchmark schema 应先转换为文档描述的标准字段，不通过猜测键名填零分。
