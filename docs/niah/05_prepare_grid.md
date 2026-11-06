# 05 固定长度×针位置测试网格

测试网格独立于训练数据生成，不对监督训练源的题目进行长度/深度重标注。默认在 held-out Paul Graham 背景中插入新 UUID；每个 trial 使用新答案，减少固定事实被模型记住而造成的误判。

```bash
bash scripts/niah/sh/21_prepare_grid.sh

# 使用与参考图接近的 10×10 长度/深度轴，另开新 grid 目录。
GRID_CONFIG=configs/niah/grid_128k.json GRID_DIR=artifacts/niah/prepared/grid_128k \
    bash scripts/niah/sh/21_prepare_grid.sh
```

## 1. 长度与深度的精确定义

横轴长度是最终 `len(input_ids)`：description + 全部背景 + needle + 完整问题 prompt。不含生成答案 token。使用当前模型 tokenizer 分别编码各部分，在 token 层面插针后直接保存 IDs，**不再次整体编码，不在插针后截断**。

设目标长度为 L，description/needle/prompt 的长度为 D/N/P，则背景预算 `B=L-D-N-P`。针的插入 offset=`round(B×depth/100)`，最终顺序：description、背景前缀、针、背景后缀、问题。0% 为所有背景之前、description 之后；100% 为所有背景之后、问题之前。针自身始终完整。

`depth_percent` 是目标深度，`actual_depth_percent=offset/B×100` 记录 token 取整后的实际值；`needle_start_token` 记录完整 encoder 序列中的绝对位置。格子按目标深度聚合，真实偏差可在每条预测中审计。这与各框架的 sentence-boundary 插入、chat-template 额外 token 或不含 prompt 的 context length 定义可能不同，报告时必须注明，不能直接比较同名长度数值。

## 2. 配置

| grid JSON 字段 | 默认 | 说明 |
| --- | --- | --- |
| corpus | paulgraham | RAW_DIR 中的背景文件名，不含扩展名 |
| task | uuid | uuid / passkey / custom；默认需要精确检索随机值 |
| lengths | [2048,4096,8192,16384] | 严格递增、无重复的完整 encoder token 长度 |
| depths | [0,10,...,100] | 严格递增、无重复、范围 0–100 |
| repeats | 5 | 每格 trial 数，不是批大小 |
| seed | 20261107 | trial seed=seed+trial；用于针与背景起点 |
| min_ratio | 4 | 计划测试的最小比例，决定最大压缩前缀预算 |
| max_decoder_tokens | 8192 | centers+完整 prompt+generation_budget 上限 |
| generation_budget | 64 | 每个样本预留的生成 token，benchmark 不得超过 |
| max_corpus_tokens | 2000000 | 背景池最大 token 数 |
| per_document_tokens | 262144 | 单篇 token 上限 |
| repeat_haystack | false | 不足长度默认报错；true 为明确的循环背景实验 |
| include_text | false | true 额外保存可读正文；IDs 始终是权威输入 |

同一 trial 的针、答案和背景起点在所有长度/深度上保持一致；较短长度使用同起点的较短窗口。trial 之间使用独立 seed。各深度配对题不是独立随机样本；每格 accuracy 的分母是 repeats，默认5意味着只能取 0/20/40/60/80/100%。最终实验需要更稳定估计时增大 repeats，并报告数量。

默认 220 条；128K 配置长度为 `[4096,18204,32312,46421,60529,74638,88746,102855,116963,131072]`，深度为 `[0,11,22,33,44,55,66,77,88,100]`，每格10次，共1000条。其解码预算扩大为40000以容纳 R=4 的较长压缩前缀。131072 表示128×1024，不是模型能在当前 GPU 上运行的承诺。四比例合计4000次生成，先单格/短网格做显存验证。

## 3. custom 针与背景选项

如需类似原始自然语言事实针，在新配置中设 `task=custom` 并添加：

```json
"custom": {
  "needle": "\n\nArchive MERIDIAN recommends visiting the orchid conservatory at dawn.\n\n",
  "question": "What activity does archive MERIDIAN recommend?",
  "answer": "visiting the orchid conservatory at dawn"
}
```

这是测试配置中的一个字段，不能替换整个 JSON。custom 的答案必须在 needle 中，生成预算必须容纳完整答案；不要用同一固定 custom 针训练又测试。固定事实可能受已有知识/记忆影响，UUID/passkey 更适合诊断确切位置检索。代码拒绝背景已经包含目标答案的样本，不尝试通过替换词语悄悄修题。

改用 PG19 test 时，先下载 pg19_test，把 corpus 改为 pg19_test；不要改为 pg19_train。使用本地背景时准备独立文本 JSONL。背景池不足最大长度默认报错，增加正文或池上限；循环重复必须显式设置并在实验报告声明。

## 4. 输出与重叠检查

输出包括 `grid.jsonl` 和 `manifest.json`。manifest 记录完整网格、实际数量、输入/深度定义、tokenizer、背景源 commit/hash、文档区间、网格文件哈希。

`--training-manifest` 可显式传入训练 manifest；省略时若默认 `artifacts/niah/prepared/train_v1/manifest.json` 存在会自动检查，否则明确记录没有训练 manifest guard。自定义 PREPARED_DIR 时应设置 TRAINING_MANIFEST 指向对应版本。构建时检查训练背景/已选答案哈希，测试时还根据当前 checkpoint 内保存的训练身份重新检查，不依赖文件名判断无重复。

该网格也可用于旧 QA 训练得到的 SeCo checkpoint 做基线；只要 tokenizer 一致并满足位置/解码预算即可。对于不含 NIAH 训练身份的旧模型，不补造其去污染保证。
