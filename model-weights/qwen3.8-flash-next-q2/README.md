# Qwen3.8-Flash-Next 混合权重：来源、获取与分发

资料日期：2026-10-10。本说明对应已经验收的 `freetoken 0.1.3+speed.q2exp1` 和 Speed Q2 实现提交 `58fbda95a40865313c75276128dc6ec53f435f58`。这里只新增权重说明和获取工具，没有重新量化模型、裁剪专家、修改引擎或发布大权重。

正式资料仓库为 [siltok-ai/BuildFreetokenForSpeed](https://github.com/siltok-ai/BuildFreetokenForSpeed)。此前私人仓库不再作为更新目标。Speed 保存运行接入及材料引用，本目录保存权重来源和分发说明。

## 1. 当前模型到底由什么组成

从直接下载来源看，是两组；从运行组成看，是三部分。

| 运行组成 | 直接下载来源 | 当前怎样使用 |
|---|---|---|
| BF16 主干、router、共享专家、embedding、LM head，以及 config/tokenizer/chat template | `RadixArk/Qwen3.8-Flash-Next-NVFP4` 的非路由专家文件 | 保留原始字节与映射；普通文本主干进入 GPU，激活 BF16 |
| FP8 PLE / n-gram lookup 及原有 scale、其他 PLE 参数 | 同一 RadixArk 仓库的 10 个 `model-plefp8-*` 分片 | 使用 FreeToken SSD PLE / Windows RowStore，稀疏读取所需行；没有替换为 GGUF 的 PLE |
| 完整 routed experts | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 的 Q2_0 第一片 | 仅读取 48 层 × gate/up/down，共 144 个专家张量；48×512 专家全部加载为 Q2_0 锁页 host bank，GPU 使用专家 cache |

“FreeToken 主干”描述的是其加载/执行方案。实际 HF 权重发布者是 RadixArk，基座是 Qwen；FreeToken 没有另提供一套我们消费的主干模型权重。

“Strata 的 Q2”是此前探索中的简称。实际专家文件发布者是 ISTA-DASLab，使用 GSQ/RCO；[Strata](https://github.com/Niko1221/Strata) 是相关 GGUF 的第三方使用入口。当前 Speed 运行 q2exp1 FreeToken，不调用 Strata 服务或加载 Strata 的运行库。量化类型是 **Q2_0**，不是 Q2_K、IQ2_XS，也不是把整个模型改成 2-bit。[量化模型卡](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/ed59f92082b1e93c0e96d60a8b11aab089b52f09/README.md)

HF 的四个 BF16 文件还保留原始视觉/MTP 等非专家张量；当前验收是纯文本路径，不声明视觉或 MTP 推理已开放。按文件保留这些张量避免本轮引入额外重切分与格式变更。

## 2. 所有原始来源及其关系

| 来源 | 地址 / 固定版本 | 与本模型的关系 |
|---|---|---|
| Qwen 基座 | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 模型架构、原始 BF16 权重和基座许可的上游；不是本轮另行完整下载的输入 |
| Qwen FP8 版本 | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | RadixArk 发布说明记录的 FP8 PLE 上游；我们直接消费 RadixArk 已发布分片，不在本轮重新转换 PLE |
| RadixArk HF checkpoint | [固定 revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) | 直接获取 14 个非专家 safetensors、原索引和 5 个模型/分词文件；不获取其 192 个 NVFP4 路由专家分片 |
| ISTA-DASLab GGUF | [固定 revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) | 直接获取 Q2_0 第一片，保留整片文件身份，只读取其中完整专家 |
| GSQ / RCO 方法 | [GSQ](https://github.com/IST-DASLab/GSQ)、[RCO](https://github.com/IST-DASLab/RCO) | 第三方量化方法来源；我们消费其已量化字节，本轮不重跑量化 |
| FreeToken 官方实现 | [FlashML-org/FreeToken，`078132480ea4a0424f677a27eb22085624d331e6`](https://github.com/FlashML-org/FreeToken/tree/078132480ea4a0424f677a27eb22085624d331e6) | 提供 Flash Next / QSA / SSD PLE / 专家 offload 的原实现 |
| 公司 q2exp1 材料 | [固定提交 `029cecd76efc7abbd72cc7484d7d8ac0d916fca5`](https://github.com/siltok-ai/BuildFreetokenForSpeed/tree/029cecd76efc7abbd72cc7484d7d8ac0d916fca5) | Windows 适配和 Q2 专家读取/计算补丁、wheel、26 DLL及验收证据；这是引擎资料，不是模型文件 |

RadixArk 模型卡说明其 routed experts 为 NVFP4、其他主干保持 BF16，PLE 使用 Qwen FP8 版本；本混合模型选择保留非专家部分，并替换路由专家。[固定 RadixArk 说明](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/blob/7b719225242aacd3dbd3f9407468c2ee9a9d2594/README.md)

**历史身份与现在固定的 revision 分开记录。** 早期 HF 下载回执记的是 `master`，没有保存不可变 commit；已经保存原索引和 14 分片的 SHA256/尺寸。本次确认上述 RadixArk revision 的 HF LFS 元数据与这些历史字节身份全部一致，5 个 metadata 文件也与本机原始来源逐字节匹配，所以将它作为可重现获取版本。不能反写成“早期下载时已锁定该 commit”。Qwen BF16/FP8 的历史父 revision 未由直接输入记录，来源锁中的 Qwen commit 仅是本次核对的说明/许可参考，不冒充已建立的逐张量父版本证明。

完整机器清单见 [source-lock.json](source-lock.json)，本次核对见 [source-verification.json](source-verification.json)。原始 SHA 回执继续在 [HF 非专家来源清单](../../integration/speed/examples/flash-next-hf-source-receipt.json)中保留。

## 3. 精确需要哪些文件

HF 侧共 20 个文件：

```text
model-bf16-00001.safetensors
model-bf16-00010.safetensors
model-bf16-00011.safetensors
model-bf16-00012.safetensors
model-plefp8-00000.safetensors ... model-plefp8-00009.safetensors
model.safetensors.index.json                 原完整索引，作为来源输入
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

不要使用 `model-bf16-*` 通配符盲下所有文件；以 source-lock 的具体名单为准。14 个大分片共 **67,208,024,363 B**。原完整索引约 34 MB，虽含未下载专家的映射，但准备工具只选择非专家键，生成运行目录的过滤索引。

专家侧只要这个文件：

```text
Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
```

文件尺寸 **37,623,740,192 B**，SHA256：

```text
69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720
```

HF 全部必要文件加 GGUF 第一片的原始 payload 为 **104,878,864,747 B（104.879 GB / 97.676 GiB）**，不含许可证/来源卡、下载缓存和暂存空间。GGUF 第一片中真正选中的专家数据为 **33,973,862,400 B（31.640625 GiB）**；下载文件大小、磁盘模型大小和专家 DRAM 占用是不同口径。

没有使用 GGUF 的主干、tokenizer、第二片 IQ4_NL PLE 或 mmproj。第一片中未消费的其他张量仍随整片下载，保留现有完整文件 SHA。这里只需第一片是 **q2exp1 专家专用 reader 的合同**；不能把它当作可在通用 llama.cpp/Strata 中独立加载的完整模型。

当前版本不应抽出 144 个张量另写一个小 GGUF：引擎与 Speed 对整片尺寸、SHA、类型及形状均有固定检查。拆除未用张量会改变身份；应视为新的模型/引擎合同，并重新验收。

## 4. 获取与构造当前已验收布局

使用固定 q2exp1 环境（不是任意当前 PyPI FreeToken），先获取本仓库；执行时固定本目录所在提交以追溯工具版本。引擎二进制仍来自上述 `029cecd...`，新增文档不会改变其身份。

```powershell
git clone git@github.com:siltok-ai/BuildFreetokenForSpeed.git
Set-Location BuildFreetokenForSpeed
git switch feat/flash-next-q2-weight-distribution

# 只查看精确下载计划，不下载权重，也不创建来源目录。
python -B model-weights/qwen3.8-flash-next-q2/download_sources.py --output D:/models/flash-next-sources --dry-run

# 已安装 huggingface_hub 的独立环境中实际下载；固定 revision、逐文件校验。
python -B model-weights/qwen3.8-flash-next-q2/download_sources.py --output D:/models/flash-next-sources
```

下载器仅获取本节清单，HF 放在 `<output>/hf`，GGUF 放在 `<output>/experts/Q2_0`。它调用既有专家下载器，使用 Hugging Face 固定 revision，最后重新读取完整 SHA。当前 ModelScope 镜像也可用，但镜像 `master` URL 只用于运输，最终字节必须等于相同固定 SHA；它不能代替不可变来源身份。

下载后，使用仓库已有准备工具：

```powershell
python -B tools/prepare_q2_checkpoint.py `
  --trunk D:/models/flash-next-sources/hf `
  --trunk-receipt integration/speed/examples/flash-next-hf-source-receipt.json `
  --experts D:/models/flash-next-sources/experts `
  --output D:/models/qwen3.8-flash-next.bf16-q2-experts

python -B tools/prepare_q2_checkpoint.py `
  --trunk D:/models/flash-next-sources/hf `
  --trunk-receipt integration/speed/examples/flash-next-hf-source-receipt.json `
  --experts D:/models/flash-next-sources/experts `
  --output D:/models/qwen3.8-flash-next.bf16-q2-experts `
  --finalize
```

必须指定显式 `--trunk-receipt`，新获取路径不依赖历史实验目录的回执。准备工具重验原索引及非专家分片、检查索引 tensor 存在和 dtype、通过硬链接复用 14 文件，复制配置/tokenizer，过滤所有 `.experts.` 索引键并新增 `freetoken_expert_source`。`--finalize` 解析全部 144 专家张量，检查 48×512 几何、GGUF type42、边界及不重叠，记录已验证文件的本地 mtime。

当前准备工具使用 `os.link`，HF 来源与输出必须处在支持硬链接的同一文件系统/卷。不同卷或不支持硬链接的存储需先复制来源到目标卷再构造；不能把软链接当成正式分发 payload。`--output` 必须是新目录，既有生产模型不在此过程中覆盖。

运行目录中有 14 个非专家文件、过滤索引、metadata 和 `mixed-source.json`；专家文件仍在 `sources/experts`，其绝对路径写入 config。源目录名虽带 NVFP4，我们实际不加载其中 NVFP4 路由专家；量化入口通过明确的 expert-source 分流。

## 5. 混合精度怎样执行

专家文件是 uint8 原生 Q2_0 块：每 64 元素有 FP16 scale 与 16 字节 2-bit code。所有真实专家序列读取并装入锁页 host bank，GPU cache 借用原通用拷贝；Q2 专家 Triton kernel 内恢复 BF16 权重，BF16 激活、中间舍入及 FP32 乘加/路由求和。主干、router、共享专家不改成 Q2。

PLE 保留 HF FP8 E4M3 数据和原有 scale，沿既有磁盘 PLE 路径读取。26 DLL 是通用已编译算子库存；Q2 计算还使用自身 Triton 编译/cache，不能把模型权重与 kernel cache 混为一套文件。

当前约束为 TP=1、纯 GPU offload、SiLU、输出侧路由加权，文本输入；默认 1024 专家槽、8192 KV tokens。主干来源、expert-source 和各自精度都有独立合同。当前 CPU/hybrid、另一量化格式、通用 GGUF 引擎或未验收平台不在这个发布身份中。

## 6. 如果公司自己分发权重，怎样打包

### 推荐首版：原始组件包 + 安装时构造运行目录

保留当前已验收字节与 reader，不重切专家 GGUF。模型大文件使用公司控制的模型托管仓库或支持 Range/续传的对象存储；本 Git 仓库保存版本合同和工具。权重单独于 Speed 安装包及引擎 wheel/cache 分发，便于升级和校验。

```text
qwen3.8-flash-next-bf16-q2exp1-model-v1/
├── MODEL-CARD.md                     公司组装版本、上游来源、适用引擎、验收边界
├── source-lock.json                  本目录来源锁
├── SHA256SUMS                       全部实际交付文件的SHA256
├── licenses/
│   ├── Qwen-Community-License-1.0.txt
│   ├── RadixArk-model-card.md
│   ├── ISTA-DASLab-model-card.md
│   └── ATTRIBUTION.md                作者、来源、方法与本地修改说明
├── hf/                              上述20个原始HF文件
├── experts/
│   └── Q2_0/
│       └── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
└── prepare-runtime说明              指向固定公司资料commit及上述准备步骤
```

这里的目录是交付设计，尚未在本轮生成或上传 104.9 GB 权重包。准备脚本来自固定公司仓库版本；离线交付还需提供脚本运行所需的 q2exp1 环境或 Speed 随包 Python/引擎及依赖闭包。

可以保持原始文件供逐文件续传；如采用容器压缩/传输分块，应另外记录每块的范围、尺寸和 SHA，恢复后再检查原文件完整 SHA。Q2/FP8 数据已经压缩，不能承诺通用压缩再大幅缩小体积。传输分块不是修改 GGUF 模型分片格式；恢复后的名字、字节和 SHA必须与来源锁一致。

使用 GB/GiB 明确标示大小。原始 payload 约97.676 GiB，安装时按源/输出同卷硬链接避免另复制14个大文件；若走复制路径，再为约62.593 GiB非专家分片预留空间。下载缓存、归档临时副本、解包及模型运行 commit 不包含在这些磁盘数字中。

**安装顺序：** 下载/恢复文件 → 对照完整校验清单 → 在安装卷使用准备工具新建运行目录 → finalize → 注册 Speed manifest → 真实推理验收。把构造后的运行目录移动到另一位置时，应重新运行这个构造流程，以生成该位置的专家引用和mtime回执；来源组件包本身可搬运。

### 直接交付准备好的运行目录为何还需要安装动作

目前 config 中的专家文件在模型目录之外，含本机绝对路径；`verified_mtime_ns` 与 `mixed-source.json.non_expert_files[*].mtime_ns` 都反映本机文件状态。跨机器复制、压缩/解压、对象存储下载可能改变mtime。

因此仅压缩现有 `qwen3.8-flash-next.bf16-q2-experts` 会漏掉 GGUF；即使补上 GGUF，也可能带着不可用路径/时间戳。不能仅把mtime改成当前值绕过来源验证，更不能删除校验字段或禁用严格准入。

引擎支持 model-root 相对 expert path，但**相对路径只解决定位，不解决mtime回执更新**。若以后要设计即装即用的紧凑运行包，可将 GGUF 放在 `weights/experts/`，在安装端重新完整SHA验证、解析专家目录、更新路径和回执，再执行完整验收；那应有独立安装工具及失败测试。本轮不引入该新安装合同，首版使用上述已存在的“组件包→本机重新构造”。

### 版本与发布记录

模型版本固定两个直接来源 revision、每文件SHA、精度、文件布局、准备工具commit、q2exp1 wheel身份与Speed适用版本。新字节、新的FP8 PLE源、重新量化、重切分或索引规则变化必须给新模型版本；不能覆盖同版本“最新权重”。

公司 MODEL-CARD 应声明“Qwen3.8-Flash-Next 混合组装版本：RadixArk 非专家 + ISTA-DASLab Q2_0 routed experts”。不要沿用“官方完整 Q2模型”“完整Strata模型”或“标准独立GGUF”的名称，也不能继承上游完整量化模型的性能/质量评测作为本组装的评测。

## 7. 许可证与公开分发条件

截至本次核对，基座采用 **Qwen Community License 1.0**。RadixArk 许可说明指向基座；固定 ISTA-DASLab 模型卡的正文也声明量化权重继承基座。GGUF卡的metadata标为Apache-2.0，与正文的继承说明不能混用，不能据该标签把整套权重宣布为Apache模型。[固定基座许可](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE)、[RadixArk许可说明](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/blob/7b719225242aacd3dbd3f9407468c2ee9a9d2594/README.md)、[GGUF继承说明](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/ed59f92082b1e93c0e96d60a8b11aab089b52f09/README.md)

发布时保留完整版权及许可声明、上游作者和组装变更说明。该许可允许分发等使用，但有条件：商业产品/服务超过1亿月活或2000万美元月收入时有显著展示模型名要求；从事商业 MaaS 或 AI Work Assistant 业务的使用者及关联方需取得Qwen另行许可，纯内部且不向第三方开放能力的使用有例外。API托管和独立办公/编码助手应按原文定义核对公司的实际业务，不能仅因改成Q2或改名就视为避开基座条款。[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE)

本说明记录发布所需核对的真实条款，不替公司认定已获得额外许可。本轮不公开上传模型文件。发布者应在权重 MODEL-CARD 和交付包中附上采用版本的许可/署名，许可参考 SHA 见source-lock。FreeToken/Windows补丁的代码许可与模型许可分别保留；没有分发Strata软件，不因此把Strata引擎许可套到本权重包。

## 8. 安装与发布验收

使用当前 Speed 的 [Flash Next接入说明](https://github.com/siltok-ai/siltok-speed/blob/feat/freetoken-flash-next-q2/silitok_speed/qwen36/FLASH-NEXT.md)和 [manifest示例](../../integration/speed/examples/flash-next-q2-experts-manifest.json)，将 `artifacts.weights.path` 指向本机新构造的运行目录，而不是原始组件 `hf/`。

先确认完整文件SHA、48×512真实专家/144张量、31.640625GiB压缩银行、无alias/INVALID，主干BF16及PLE路径不变。之后在目标机器实际验证短问题、完整逻辑题、长输入、thinking、HTTP/SSE、取消后复用、卸载及旧模型能力。资源检查必须区分物理RAM、VRAM和Windows commit；host36/VRAM14预算及专用safety是既有验收条件，不能把本机低余量配置推广成任何机器都能运行。

既有正确性证据、随包测试和来源数值验证见 [q2exp1候选说明](../../docs/q2-expert-candidate.md)；性能见 [Speed性能报告](https://github.com/siltok-ai/siltok-speed/blob/edb67b273ddba141c78287eb86d5fd759d2e6b23/doc/speed/models/qwen38-flash-next-q2-performance-20261010.md)。复制了相同字节不等于已完成新发布的安装器/干净机器验收；本轮未复做GPU测试或生成新的公开权重版本。

## 9. 本目录提供什么

- [source-lock.json](source-lock.json)：固定直接来源、全部文件尺寸/SHA、间接上游与引擎身份。
- [source-verification.json](source-verification.json)：14分片LFS身份及原索引、metadata历史字节匹配结果。
- [download_sources.py](download_sources.py)：只下载需要的原始组件，支持不创建目录的dry-run；大文件下载路径未经本轮重新跑完整97.7GiB下载。
- 现有准备/专家下载工具位于仓库 `tools/`，不修改已冻结的引擎补丁、wheel或cache。
