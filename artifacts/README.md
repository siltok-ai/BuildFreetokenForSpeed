# FreeToken 构建材料

当前材料由 [current-release.json](../integration/speed/current-release.json) 选择：cloverq8-2-cp312-cu130-sm120。

- wheels/：CPython 3.12 Windows x64 wheel，含 pinned tensor、CPU MoE、row store、GGUF 四个原生模块。
- kernel-cache/：34项AOT DLL，完整规格以匹配的release合同为准。
- build-record.json：源码输入、版本、工具链、wheel及各DLL实际SHA256。

wheel和AOT cache必须来自同一材料修订。运行 tools/verify_materials.py 核对实际文件；历史修订可从Git获取，不覆盖旧二进制。构建步骤见 [build](../build/README.md)。
