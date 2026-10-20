# 仓库工具

从仓库根执行 Python 入口，使用 Python 3.12 和 Git。

| 入口 | 用途 |
| --- | --- |
| `prepare_source.py` | 按当前 profile 重放源码到新目录 |
| `validate_repository.py` | 校验官方快照、当前源码及材料 |
| `verify_current_release.py` / `verify_portable_consumer.py` | 当前材料及可选 Speed 合同兼容性 |
| `verify_materials.py` / `validate_python_revision.py` | 独立输出库存及 wheel 源码检查 |
| `flash_next_weights.py` | 固定 GGUF 来源的下载、准备与检查 |
| `flash_next_mtp.py` | 官方 MTP 下载、转换和检查 |
| `flash_next_bundle.py` | 制作完整、可搬迁的资源包 |
| `flash_next_speed_config.py` | 为独立模型目录生成单模式配置 |
| `export_upstream.py` | 从明确上游 commit 导出源码候选 |

模型工具依赖与操作步骤见 [资源包说明](../docs/flash-next-gguf-bundle.md)，完整编译入口在 [build](../build/README.md)。工具测试安装 `requirements-test.txt`；离线仓库检查不需要 CUDA。

`download_ranges.py` 是 GGUF 下载实现，`flash_next_gguf.py` 负责读取和导出 GGUF 元数据。`mtp/` 包含 MTP 来源获取与转换实现；其 [references](mtp/README.md) 保留实际使用的第三方函数及许可。运行输出和探索记录放在仓库外。
