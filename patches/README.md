# 本地差异的入口

机器按 [series.json](series.json) 的顺序应用补丁；人按 [改动说明](../docs/changes.md) 理解每组的理由。编号 FTWIN / FTEXP / FTPKG 是后续重放、退休和追踪的稳定标识。

| 顺序 | 稳定编号 | 差异 |
| --- | --- | --- |
| 1 | FTWIN-001 | [Windows 运行与 SSD IO](series/0001-windows-runtime.patch) |
| 2 | FTWIN-002 | [MSVC / CUDA 构建](series/0002-windows-build.patch) |
| 3 | FTEXP-001 | [INVALID 专家复用实验](series/0003-invalid-expert-experiment.patch) |
| 4 | FTPKG-001 | [定制 wheel 版本](series/0004-speed-wheel-version.patch) |

这四组按文件边界从此前实际构建使用的 [完整补丁](legacy/windows-source.patch) 拆分，29 个修改文件没有重叠。仓库验证器实际应用两种表示，并比较所有文件，保证结果完全相同。

`legacy/` 是历史溯源证据，新构建使用 `series.json`。后续上游升级的补丁自然可能不同于 win1 历史补丁：应为新版本建立自己的 series / profile / 验证规则，保留 win1，而不是改写历史等价断言或旧 SHA。

仅删除实验补丁不能直接称为当前 win1 等价版本。正式无实验代码版本应采用新版本号、重建 wheel 与 cache，再执行旧模型回归。
