# MTP 来源获取与转换

公共入口是 `python -B tools/flash_next_mtp.py <plan|fetch|build|verify> --root <输入根>`。完整步骤见 [资源包说明](../../docs/flash-next-gguf-bundle.md)。

- `fetch_source.py` 从固定 Qwen revision 获取 31 份 BF16 张量。
- `build_runtime.py` 流式转换专家及其他张量，输出 31 个紧凑数据文件和固定 runtime 清单。
- `references/` 保留 Strata 固定 commit 的下载器、Q2/Q8 量化函数所在原文件及 MIT 许可证。转换器只从 AST 提取所需量化函数，不运行参考文件的 CLI，也不依赖 Strata 引擎或 gguf-py。

参考来源、文件身份记录在 `references/origins.json`。移动本目录不会改变转换格式、张量来源或生成数据。

`runtime.json` 使用固定发布清单，其中 `status: packed_unvalidated_runtime` 和 `engine_integrated: false` 保留的是生成清单时的记录，不表示当前 Speed 的加载结果。`verify` 检查 31 份运行数据；实际推理另做 OFF/ON 验证。本次转换的诊断和耗时在 `build` 输出目录的 `conversion-report.json`，不必修改发布清单中的状态或时间。
