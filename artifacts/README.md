# FreeToken 构建材料

当前可交付材料读取 [current-release.json](../integration/speed/current-release.json) 的 `materials`。当前目录为 `strataq2portable1-cp312-cu130-sm120`：

- `wheels/`：CPython 3.12 Windows x64 wheel，包含 CPU MoE、pinned tensor、row store 三个原生模块。
- `kernel-cache/`：SM120 的 27 项 AOT DLL、构建元数据及 Speed 缓存清单。
- `build-record.json`：版本、源码输入、工具链和实际材料库存。

将整个材料目录作为一套使用。Speed 打包时显式提供 wheel 和 kernel-cache，见 [接入说明](../docs/speed-integration.md)。

```powershell
python -B tools/verify_current_release.py
python -B tools/verify_materials.py D:\materials\freetoken
```

`strataq2perf9-cp312-cu130-sm120` 是当前版本原生二进制复用所需的基线，保留它供源码差异检查及重新封装使用；Speed 消费当前 portable1 目录。其更早的来源记录可从 Git 历史查询，日常检查不递归验证历史版本。

新构建输出到仓库外的新目录，经验证后再决定是否发布；不要覆盖现有 wheel/DLL。完整构建与 Python 修订重新封装见 [build](../build/README.md)。本目录不保存性能探索或安装验收日志。
