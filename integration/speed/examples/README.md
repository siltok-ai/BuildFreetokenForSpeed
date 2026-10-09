# Manifest使用范围

两份manifest模板使用相对路径，复制到独立models目录后，应在其`weights/qwen36`、`weights/qwen38`、`weights/flash-next`位置放置权重目录或NTFS junction；不要修改正常下载的原始模型索引。模板不包含权重。

这些模板只包含配置，不包含模型权重或本机目录。目标机器需将相对权重路径绑定到自己的模型位置；Windows 可使用 NTFS 目录 junction。

普通两模型保持正常合同。Flash Next条目必须保留`INVALID-alias20`标识和两个显式engine.extra开关。当前权重目录保留30层专家，Speed实验只分配20物理层，执行48逻辑层；PLE放SSD。输出不能用于正确答案或模型质量对比。

本机HTTP验收分别显式采用：Qwen3.6全局VRAM/commit余量2GiB/8GiB；Qwen3.8-27B为0.75GiB/8GiB；Flash Next实验为2GiB/1GiB，并使用独立32GiB host、16GiB VRAM预算。模板的模型参数不包含全局资源安全余量；接入自己的Engine服务时需要根据机器空闲量配置它们，不能假设默认余量均能准入。完整Flash Next正常profile仍要求补齐缺失专家分片和足够RAM/commit。

配套DLL目录通过`service.freetoken_kernel_cache_dir`配置；若从venv安装wheel使用其site-packages，不需要源码覆盖。Speed 仓库中对应的探索工具使用 `--profile qwen36`、`--profile qwen38`、`--profile flash-next`；它属于 Speed 源码，不由本仓库提供。Flash Next实验验收还显式使用 `--invalid-expert-keep-layers 20 --host-commit-estimate-gib 32 --vram-estimate-gib 16 --commit-safety-gib 1`。

版本、文件身份、打包参数和正常/实验 worker 边界见 [接入约定](../../../docs/speed-integration.md)。
