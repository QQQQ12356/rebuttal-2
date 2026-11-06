# 08 NIAH 验证记录与排错

## 1. 当前已做与尚未做

本次 Windows 中仅编辑和静态检查，未执行真实数据下载、预处理、训练、GPU 推理或最终绘图。未生成任何虚构准确度。已做检查包括新 Python AST、配置 JSON、Shell bash -n/LF、独立入口文件存在性；原始 SeCo 两段压缩方法的 AST 一致性仍通过。

当前 Windows 运行时没有 Ruff，完整 Ruff 和以下 Linux CPU 用例待目标环境运行。代码已实现不代表 DeepSpeed/CUDA组合或128K显存已经验证。

## 2. Linux 无 GPU 验证

```bash
python -m pip install -r requirements-niah.txt -r requirements-tools.txt
bash scripts/niah/sh/90_validate.sh
```

验证使用临时字符 tokenizer、临时正文、合成分数，不下载模型或数据：

| 测试内容 | 能发现的问题 |
| --- | --- |
| 固定输入长度、0/50/100%插入、完整针 token | 长度预算、端点和插针后截断错误 |
| 训练/测试 namespace | 相同 seed 误用同一答案 |
| 背景不足默认失败 | 静默循环背景 |
| 监督源重新分词/拒绝 test split | 混用其他 tokenizer IDs 或把测试题训练 |
| accuracy 单格重复数 | 把0/1与1–10或百分比混淆 |
| 重复/漏测 grid 检查 | 分片补齐重复、缺失格子被补零 |
| 数字边界 hit 检查 | 部分数字子串假阳性 |
| 临时 PNG/PDF/CSV 导出 | headless绘图、输出协议、逐格值一致性 |

这些是已写、待 Linux 执行的用例。图像测试产生的是临时 toy 图，不能用作论文模型成绩。

## 3. 建议的 GPU smoke test

1. 创建新的小 prepare_train 配置，synthetic 32条、每监督源16条，输出 train_smoke；保留合法长度预算。
2. 创建新 grid_smoke 配置，lengths=[2048,4096]、depths=[0,50,100]、repeats=2，输出 grid_smoke，共12条；显式关联训练 manifest。
3. 复制 NIAH train.json，设置 max_steps=2、save_steps=1、logging_steps=1，先单卡 DDP 训练；确认 finite loss、checkpoint-1/2/final。
4. 在 checkpoint-1 同 backend/world size 恢复到第2步，检查 optimizer、scheduler、global_step 和数据身份。
5. 新 run 分别验证多卡 DDP、ZeRO-2、CPU 卸载；不要以单卡成功代替 ZeRO 保存恢复验证。
6. 合并 final，只测 --ratios 16，确认12个预测、6格且每格2条；比较 adapter --no-merge 与 merged 结果。
7. 使用相同 grid 和模型对比单卡/多卡 ID集合、EM/F1，再作图抽查CSV中一格与预测结果相同。
8. 扩大长度/重复数/比例；128K从单格、batch1开始，不能直接拿参考图作为显存预算。

第一次使用自定义 TRAINING_MANIFEST 时核对其指向当前 NIAH train 数据版本。后续模型对比应固定同一 grid，不因模型结果变化更改样本、问题、答案或深度。

## 4. 排错

| 报错/现象 | 处理 |
| --- | --- |
| Expected niah training data / Expected qa training data | 入口与 prepared task 混用，换独立入口和目录 |
| No eligible training examples | 查 long_input/long_answer/long_decoder、候选范围与 tokenizer；不能截断针来凑数 |
| 背景池不足 | 下载更多背景或提高池/单篇上限；重复背景必须显式开启并报告 |
| Needle answer already occurs | 换新测试 seed/独立窗口，或检查 custom 答案是否过于常见 |
| train/test overlap | 检查 corpus split、训练背景和答案重复；不能关闭检查后继续当作 held-out 结果 |
| Tokenizer differs | 网格与 checkpoint 的模型/tokenizer不同；重新构造网格 |
| decoder/generation budget exceeded | 提高对应 grid 预算并重新构建，或减少生成长度/降低中心数 |
| Model position limit exceeded | 使用支持该输入长度的真实基座；不要忽略位置上限警告 |
| CUDA OOM | batch降为1，减长度、减余弦分块大小；测试多卡只是数据并行，不拆一份模型 |
| Incomplete/duplicate grid | 检查 rank首个报错、分片数量、trial；默认不发布完整指标 |
| 没有 metrics 但有 rank预测 | 测试中途失败；保留日志后用新输出重跑，部分计分必须明确 allow-incomplete |
| 绘图输出已存在 | 换 FIGURE_PATH，避免覆盖已有实验图与CSV |
| 多面板 grid身份不一致 | 使用同一 prepared grid 的结果；不要只对齐轴标签后拼图 |
| 色条看似与参考图不同 | 当前为EM百分比，参考图1–10来自另一评估方式；不能按视觉强行换分数 |

关键代码核对位置：`seco/niah/data.py` 构造长度/位置，`seco/niah/metrics.py` 验证网格与计分，`seco/niah/plotting.py` 固定色标和缺失掩码，`seco/train_cli.py` 共享训练与数据任务校验。原始压缩方法始终在 `seco/compression.py`。
