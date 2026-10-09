# 制品与证据

[win1-cp312-cu130-sm120](win1-cp312-cu130-sm120/) 是本机实际验证过的初始制品集：一个 wheel、26 DLL、cache 合同、构建身份与报告。不是某次 GitHub CI 自动编译结果，也不是完整 Speed 安装器。

复用时复制完整子目录并运行 `python -B tools/verify_materials.py <复制后的目录>`。不要从不同版本目录挑选 DLL 混用。具体 Speed 消费方式见 [接入约定](../docs/speed-integration.md)。

## 保留历史真实性

`build-record.json`、cache 元数据与验证报告保留原始内容。原工具包当时叫 `source/base.zip`、`source/windows-source.patch`、`speed-specs.txt`，历史 JSON 中仍使用这些名称；它们在本仓库分别对应：

| 历史名称 | 新位置 |
| --- | --- |
| `source/base.zip` | `upstream/freetoken-0781324.zip` |
| `source/windows-source.patch` | `patches/legacy/windows-source.patch` |
| `speed-specs.txt` | `build/specs/speed-sm120.txt` |

校验器验证映射后的 SHA 一致，并验证拆分补丁结果一致。历史日志里的本机路径只用于定位当时证据，构建入口不依赖这些路径。

历史 AOT 是 25 项整批重建，再用同一冻结源码补构 dense KV store，共 26 项。当前规格已经包含完整 26 项，新入口可一次生成；不把历史记录改写成当时已经使用新目录、新脚本一次编完。

新构建会输出新 `build-record.json`，使用本仓库路径、实际 producer SHA 与 `built_requires_runtime_validation` 状态。重编译的 SHA 可能变化，不能覆盖这份历史已验证结果。长期多版本制品转向 Releases，参见 [维护方案](../docs/maintenance.md)。
