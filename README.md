# BuildFreetokenForSpeed

为 [Speed](https://github.com/siltok-ai/siltok-speed) 提供 FreeToken 定制补丁、Windows wheel/AOT 和 Flash Next 模型资源包制作工具。Speed 的服务、授权、HTTP 和安装器逻辑在 Speed 仓库维护。

当前版本由 [current-release.json](integration/speed/current-release.json) 选择：`0.1.3+speed.strataq2portable1`，Windows x64、CPython 3.12、SM120，Torch 2.11.0+cu130。材料包含一个 wheel（3 个原生模块）和 27 项 AOT DLL。

## 获取与使用

```powershell
git clone https://github.com/siltok-ai/BuildFreetokenForSpeed.git
cd BuildFreetokenForSpeed
```

| 需要完成的事情 | 入口 |
| --- | --- |
| 给 Speed 开发环境或 Windows 安装包提供 wheel/AOT | [Speed 接入](docs/speed-integration.md) |
| 制作包含全部 Flash Next 模型资源的可搬迁目录 | [资源包制作与部署](docs/flash-next-gguf-bundle.md) |
| 从源码构建 wheel 和全部 AOT | [构建说明](build/README.md) |
| 维护 FreeToken 定制补丁或更新上游 | [补丁说明](patches/README.md)、[维护规则](docs/maintenance.md) |

使用前将本仓库所选版本与 Speed 的 `silitok_speed/runtimes/freetoken/release.json` 对齐。当前 `main` 配套 Speed 的 `feat/flash-next`；其他 Speed 版本按其 release 选择对应材料版本。

Flash Next 优先使用 **siltok31 上已处理好的资源包**：完整复制该机器的 `C:\workspace\flash-next-resources` 目录，配套 Speed 即可使用。不方便获取或希望自行制作时，按 [资源包流程](docs/flash-next-gguf-bundle.md) 从固定来源生成内容相同的包。

主模型及配置、tokenizer、模板来自同一来源的完整两片 Q2 GGUF；官方 Qwen MTP 预先转换并放入同一资源包。接收方无需运行转换工具。

## 仓库检查

工具测试使用 Python 3.12 和 Git，不需要 CUDA 或完整 Speed 环境：

```powershell
python -m pip install -r tools/requirements-test.txt
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

检查覆盖当前补丁重放、wheel 源码、原生材料和模型准备工具。GPU 推理与安装器验收分别进行，离线检查不授予实机资格。

默认模型预算适用于 RTX 5070 Ti 16GiB，建议主机内存 64GiB。MTP 默认关闭，推测解码使用 `temperature=0`。其他设备需调整预算；已知间歇长输出异常见 [资源包说明](docs/flash-next-gguf-bundle.md)。

官方源码在 `upstream/FreeToken` 保持原样；权重大文件及运行日志放在仓库外。许可及第三方署名见 [NOTICE](NOTICE)。
