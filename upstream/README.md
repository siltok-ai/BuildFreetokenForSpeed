# 官方源码输入

来源：[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken/tree/078132480ea4a0424f677a27eb22085624d331e6)。固定完整 commit 为 `078132480ea4a0424f677a27eb22085624d331e6`，源码内版本为 `0.1.3`。

| 内容 | 角色 |
| --- | --- |
| [lock.json](lock.json) | 记录来源、commit、归档 SHA、浏览目录和许可证 |
| [freetoken-0781324.zip](freetoken-0781324.zip) | `git archive` 导出的冻结源码；构建从此解压 |
| [FreeToken/](FreeToken/) | 同一归档未经修改的展开副本，供人类阅读 |

归档 SHA-256：`49a51eb5480c61571a69b44684f236cd86992b2b79062544f85bbcd263fd2443`。归档包含 686 个文件，另含目录条目；两个源码表示由仓库验证器逐文件比较。

## 官方代码导航

- [models/qwen4_exp](FreeToken/python/freetoken/models/qwen4_exp/)：Flash Next 架构、配置解析、权重加载与 PLE。
- [qwen4_exp/ple_disk.py](FreeToken/python/freetoken/models/qwen4_exp/ple_disk.py)：官方磁盘 PLE 数据源。
- [models/loader.py](FreeToken/python/freetoken/models/loader.py)：统一加载与 runtime 构建。
- [moe/expert_banks.py](FreeToken/python/freetoken/moe/expert_banks.py)、[moe/host_banks.py](FreeToken/python/freetoken/moe/host_banks.py)：专家存储、主机内存与 offload。
- [models/nvfp4_banks.py](FreeToken/python/freetoken/models/nvfp4_banks.py)：NVFP4 专家读取。
- [kernel/csrc/row_store](FreeToken/python/freetoken/kernel/csrc/row_store/)：磁盘行读取原生扩展。
- [setup.py](FreeToken/setup.py)、[pyproject.toml](FreeToken/pyproject.toml)：官方包构建与依赖声明。

此目录中的官方 README、AGENTS 和 CONTRIBUTING 说明原项目的约定。这里是只读来源快照，不是我们的定制工作目录。修改应发生在补丁应用后的新目录，最终保存到 `patches/series/`。

下一次更新用 [export_upstream.py](../tools/export_upstream.py) 从明确 commit 导出候选，再按 [维护流程](../docs/maintenance.md) 审核，不自动追随 main。
