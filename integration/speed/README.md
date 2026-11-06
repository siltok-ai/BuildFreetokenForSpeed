# Speed 消费入口

[current-release.json](current-release.json) 是本仓库当前材料的统一选择，包含版本、profile、材料目录、模型来源、MTP 清单及设备 preset。

[consumer-contract-cloverq8.json](consumer-contract-cloverq8.json) 描述当前完整构建材料的兼容要求，由 `current-release.json` 统一选择。以下命令将它与实际消费方 Speed 的 `release.json` 配对；记录实际检出的两仓修订，不以分支名称判断兼容。检查验证源码、材料及运行时合同，不执行 GPU 推理。

```powershell
python -B tools/verify_current_release.py
python -B tools/verify_portable_consumer.py --speed-root D:\src\siltok-speed
```

Speed 侧也可执行 `python -B tools/verify_flash_next_checkpoint.py --build-root <本仓库>`。兼容性按版本、ABI、Torch/CUDA、原生模块、AOT 规格和 MTP 布局判断。

[Flash Next preset](presets/flash-next-sm120-auto-cloverq8.json) 保留原有OFF/ON运行参数，只更新材料版本；[Clover preset](presets/clover-q8-sm120-auto.json) 提供独立16GiB/32GiB档位。资源包声明 `engine.resource_profile: auto`，由 Speed 选择档位。

- [开发环境及安装包接入](../../docs/speed-integration.md)
- [Flash Next 资源包制作与部署](../../docs/flash-next-gguf-bundle.md)
- [Clover Q8 资源包](../../docs/clover-q8.md)
