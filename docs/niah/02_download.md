# 02 NIAH 数据来源与下载

## 1. 本次选择的数据

本项目组合专门的检索监督训练源与可控合成样本，不将外部 benchmark 的测试题直接用作训练。

| 名称 | 来源与角色 | 在本项目中的处理 |
| --- | --- | --- |
| cogito_bits | [ksopyla/cogito-probe-bits](https://huggingface.co/datasets/ksopyla/cogito-probe-bits)，train split | 长背景中的 key-value 查询；使用 context/query/answer 文本重新分词 |
| cogito_bind | [ksopyla/cogito-probe-bind](https://huggingface.co/datasets/ksopyla/cogito-probe-bind)，train split | 实体属性/关系检索监督；独立 train split，与 Bits 一同作为补充训练 |
| pg19_train | [emozilla/pg19](https://huggingface.co/datasets/emozilla/pg19)，train split | 长篇正文背景，不自带 NIAH 标签；本地插入新 passkey/UUID 构造训练题 |
| paulgraham | [gkamradt 仓库中的 essays](https://github.com/gkamradt/needle-in-a-haystack/tree/main/needlehaystack/PaulGrahamEssays) | 仅作为 held-out 测试背景，接近指定参考实现的背景类型 |
| pg19_test | emozilla/pg19，test split | 可选 held-out 背景，默认不下载；更换 grid.corpus 后启用 |

Cogito 数据不是原始 gkamradt 单针深度网格：Bits 含 packed multi-key recall，Bind 含实体绑定任务。它们用于额外训练，不把其结果冒充单针基准。公开卡片提供 train/validation/test，下载器只读取 train；卡片现有分词是其他模型的 tokenizer，**不能直接给 Qwen 使用**。

还参考了 [NVIDIA RULER 的 NIAH 任务设计](https://github.com/NVIDIA/RULER/blob/main/scripts/synthetic.yaml)：正文/噪声背景、数字或 UUID 值。当前代码生成的是本项目 SeCo 的明确版本协议，不声称逐字复现完整 RULER，也不依赖运行其生成脚本。

## 2. 下载

```bash
bash scripts/niah/sh/10_download_data.sh

# 可选 held-out PG19 test；不会下载或覆盖五数据集流程。
bash scripts/niah/sh/10_download_data.sh --sources pg19_test

# 只下载一个源到新 raw 版本。
RAW_DIR=artifacts/niah/raw_v2 bash scripts/niah/sh/10_download_data.sh --sources cogito_bits
```

| Python 参数 | 默认 | 含义 |
| --- | --- | --- |
| --registry | configs/niah/sources.json | 明确列出各源 repo/config/split/revision/role |
| --output | artifacts/niah/raw | 独立 raw 根目录 |
| --sources | 注册表 enabled 的源 | 一次可指定多个；pg19_test 默认 enabled=false |
| --limit | 使用各源 limit | 覆盖每源读取前 N 行；0 表示全部，不是随机抽样 |

注册表字段：`kind=hf/github` 指下载方式；`role=haystack_train/haystack_test/supervised_train` 指用途；`revision` 可改 commit；HF 的 `config/split` 必须真实存在。训练用途强制 `split=train`。`fields` 仅保存所需文本和元数据，省去源数据大量不兼容 token IDs。默认 PG19 train 取前 64 篇，两个 Cogito 源扫描完整 train，Paul Graham 读取仓库该目录全部 `.txt`。

每源生成 `<name>.jsonl` 和 `<name>.source.json`。后者含完整注册表项、解析后的 commit、行数和 SHA256。相同身份/哈希会复用；不同版本或不完整文件拒绝覆盖，换 RAW_DIR 重试。限行参数选的是源前缀，后续训练预处理才对合格候选均匀抽样；报告训练数据时应说明候选范围。

## 3. 镜像、本地文件与来源追溯

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/niah/sh/10_download_data.sh \
    --sources pg19_train cogito_bits cogito_bind
```

HF_ENDPOINT 只作用于 Hub 请求。Paul Graham 从 GitHub 解析 commit，再读取该 commit 的 archive；只在内存读取目标文本文件，不向磁盘提取 archive 路径。GitHub 网络或公开 API 速率限制失败时可稍后重试，或在 Linux 自行准备合法来源的本地文件。

本地背景格式：`{"id":"文档标识","text":"正文"}`，存为 `RAW_DIR/<corpus>.jsonl`。本地监督格式至少为 `{"id":"样本标识","context":"正文","query":"问题","answer":"答案"}`；若带 split/source_split，必须为 train。没有 sidecar 的输入会标明 local_file 与实际哈希，不补造 Hub commit。

模型下载使用 `scripts/niah/sh/01_download_model.sh`，默认保存到此前同一默认模型目录以复用文件。`MODEL_ID/MODEL_PATH/REVISION` 可分别改仓库、目录、版本；改变 tokenizer 后重新生成 NIAH 训练集和网格。

## 4. 与五数据集下载修复的适用范围

本页的独立 NIAH 下载器仍通过自己的 HF 流式读取和 GitHub archive 实现获取原始数据。五数据集入口新增的 DOWNLOAD_MODE=local、SOURCE_CACHE、分片重试参数未接入此 NIAH 入口；不要把变量传给 NIAH Shell 后误以为其下载路径已改变。HF_HOME/HF_ENDPOINT 仍按本页公共规则生效。

若 NIAH 下载也遇到网络断连，应保留完整报错，按本页说明准备可追溯的本地 raw JSONL，或修复对应源的网络访问。五数据集的具体修复行为和验证命令见 [Benchmark 下载指南](../benchmark/02_download.md#5-incompleteread--chunkedencodingerror-修复)。
