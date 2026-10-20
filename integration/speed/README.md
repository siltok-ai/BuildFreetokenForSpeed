# Speed 消费入口

[current-release.json](current-release.json) 是本仓库当前材料的统一选择，包含版本、profile、材料目录、模型来源、MTP 清单及设备 preset。

[consumer-contract-strataq2portable1.json](consumer-contract-strataq2portable1.json) 描述当前材料的兼容要求。Speed 的 `release.json` 继续使用这个路径和 `tools/verify_portable_consumer.py`；该检查验证源码、材料及运行时合同，不读取历次实机报告，也不宣称执行了 GPU 推理。

```powershell
python -B tools/verify_current_release.py
python -B tools/verify_portable_consumer.py --speed-root D:\src\siltok-speed
```

Speed 侧也可执行 `python -B tools/verify_flash_next_checkpoint.py --build-root <本仓库>`。兼容性按版本、ABI、Torch/CUDA、原生模块、AOT 规格和 MTP 布局判断。

[presets/flash-next-sm120-16g-portable1.json](presets/flash-next-sm120-16g-portable1.json) 是 RTX 5070 Ti 16GiB 的资源预算，供资源包生成器使用。其他设备应单独配置和验证。

- [开发环境及安装包接入](../../docs/speed-integration.md)
- [Flash Next 资源包制作与部署](../../docs/flash-next-gguf-bundle.md)
