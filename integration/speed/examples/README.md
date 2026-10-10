# Manifest使用范围

当前 Q2_0 候选使用 [flash-next-q2-experts-manifest.json](flash-next-q2-experts-manifest.json)，对应完整 HF 主干 + 全部 Q2_0 专家，并使用 q2exp1 wheel/cache。将 `weights/qwen3.8-flash-next.bf16-q2-experts` 绑定到实际混合模型目录；新 Speed 分支不支持专家复用实验。原始非专家文件的独立校验清单为 [flash-next-hf-source-receipt.json](flash-next-hf-source-receipt.json)，供 checkpoint 准备工具重新 hash 原始 HF 文件使用。

默认候选预算为 36 GiB host / 14 GiB VRAM；本机 HTTP 实测使用 commit safety 1 GiB / VRAM safety 2 GiB。安全余量属于服务全局配置，manifest 不降低全局默认值。复现与实际证据见 [Q2_0 说明](../../../docs/q2-expert-candidate.md)。

## 历史 win1 模板

以下说明及 INVALID 模板只适用于封存的 win1/旧 Speed 分支，不是当前 Q2_0 候选入口。

两份manifest模板使用相对路径，复制到独立models目录后，应在其`weights/qwen36`、`weights/qwen38`、`weights/flash-next`位置放置权重目录或NTFS junction；不要修改正常下载的原始模型索引。模板不包含权重。

这些模板只包含配置，不包含模型权重或本机目录。目标机器需将相对权重路径绑定到自己的模型位置；Windows 可使用 NTFS 目录 junction。

普通两模型保持正常合同。Flash Next条目必须保留`INVALID-alias20`标识和两个显式engine.extra开关。当前权重目录保留30层专家，Speed实验只分配20物理层，执行48逻辑层；PLE放SSD。输出不能用于正确答案或模型质量对比。

本机HTTP验收分别显式采用：Qwen3.6全局VRAM/commit余量2GiB/8GiB；Qwen3.8-27B为0.75GiB/8GiB；Flash Next实验为2GiB/1GiB，并使用独立32GiB host、16GiB VRAM预算。模板的模型参数不包含全局资源安全余量；接入自己的Engine服务时需要根据机器空闲量配置它们，不能假设默认余量均能准入。完整Flash Next正常profile仍要求补齐缺失专家分片和足够RAM/commit。

配套DLL目录通过`service.freetoken_kernel_cache_dir`配置；若从venv安装wheel使用其site-packages，不需要源码覆盖。Speed 仓库中对应的探索工具使用 `--profile qwen36`、`--profile qwen38`、`--profile flash-next`；它属于 Speed 源码，不由本仓库提供。Flash Next实验验收还显式使用 `--invalid-expert-keep-layers 20 --host-commit-estimate-gib 32 --vram-estimate-gib 16 --commit-safety-gib 1`。

版本、文件身份、打包参数和正常/实验 worker 边界见 [接入约定](../../../docs/speed-integration.md)。
