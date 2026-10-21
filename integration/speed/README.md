# Speed 消费入口

[current-release.json](current-release.json) 是本仓库当前材料的统一选择，包含版本、profile、材料目录、模型来源、MTP 清单及设备 preset。

[consumer-contract-mtpdecode2.json](consumer-contract-mtpdecode2.json) 描述当前材料的兼容要求，由 `current-release.json` 统一选择。以下命令将它与实际消费方 Speed 的 `release.json` 配对，不要求 Speed 使用特定分支；该检查验证源码、材料及运行时合同，不执行 GPU 推理。

```powershell
python -B tools/verify_current_release.py
python -B tools/verify_portable_consumer.py --speed-root D:\src\siltok-speed
```

Speed 侧也可执行 `python -B tools/verify_flash_next_checkpoint.py --build-root <本仓库>`。兼容性按版本、ABI、Torch/CUDA、原生模块、AOT 规格和 MTP 布局判断。

[presets/flash-next-sm120-auto.json](presets/flash-next-sm120-auto.json) 是当前资源包生成器的默认配置，声明 `engine.resource_profile: auto`，由 Speed 选择现有的 16GiB/32GiB 资源档位，避免把制包机器的专家容量固定到其他机器。

- [开发环境及安装包接入](../../docs/speed-integration.md)
- [Flash Next 资源包制作与部署](../../docs/flash-next-gguf-bundle.md)
