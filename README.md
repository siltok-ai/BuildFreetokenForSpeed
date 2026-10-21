# BuildFreetokenForSpeed

为 [Speed](https://github.com/siltok-ai/siltok-speed) 提供 FreeToken 定制补丁、Windows wheel/AOT 和 Flash Next 模型资源包制作工具。Speed 的服务、授权、HTTP 和安装器逻辑在 Speed 仓库维护。

当前版本由 [current-release.json](integration/speed/current-release.json) 选择：`0.1.3+speed.mtpdecode2`，Windows x64、CPython 3.12、SM120，Torch 2.11.0+cu130。材料包含一个 wheel（3 个原生模块）和 27 项 AOT DLL。

`main` 维护当前交付，默认材料、构建入口和消费校验均选择上述版本。Speed 无论使用功能分支、dev、main 或 release，只要其 release 合同匹配，都从本仓库 `main` 取得配套输入。可直接消费仓内材料，或按[构建步骤](build/README.md)在新目录重建。

## 获取与使用

```powershell
git clone --branch main https://github.com/siltok-ai/BuildFreetokenForSpeed.git
cd BuildFreetokenForSpeed
```

| 需要完成的事情 | 入口 |
| --- | --- |
| 给 Speed 开发环境或 Windows 安装包提供 wheel/AOT | [Speed 接入](docs/speed-integration.md) |
| 制作包含全部 Flash Next 模型资源的可搬迁目录 | [资源包制作与部署](docs/flash-next-gguf-bundle.md) |
| 从源码构建 wheel 和全部 AOT | [构建说明](build/README.md) |
| 维护 FreeToken 定制补丁或更新上游 | [补丁说明](patches/README.md)、[维护规则](docs/maintenance.md) |

使用前将本仓库当前版本与实际消费方 Speed 的 `silitok_speed/runtimes/freetoken/release.json` 对齐。兼容性由版本、ABI、依赖和 AOT 合同决定，不由 Speed 分支名称决定；不匹配时先同步配套代码和材料。

按[资源包流程](docs/flash-next-gguf-bundle.md)从固定公开来源制作 Flash Next 整合包，不需要访问其他机器的目录。已有按同一流程生成的完整资源包时，也可直接复制使用。

主模型及配置、tokenizer、模板来自同一来源的完整两片 Q2 GGUF；官方 Qwen MTP 预先转换并放入同一资源包。接收方无需运行转换工具。

## 仓库检查

工具测试使用 Python 3.12 和 Git，不需要 CUDA 或完整 Speed 环境：

```powershell
python -m pip install -r tools/requirements-test.txt
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

检查覆盖当前补丁重放、wheel 源码、原生材料和模型准备工具。GPU 推理与安装器验收分别进行，离线检查不授予实机资格。

新生成的配置使用 Speed 已有的自动16GiB/32GiB档位；5070 Ti 使用较小专家池，90的大显存档位不变，建议主机内存64GiB。MTP 默认关闭，推测解码使用 `temperature=0`。旧包权重和MTP数据可复用，固定部署参数的升级及使用限制见 [资源包说明](docs/flash-next-gguf-bundle.md)。

官方源码在 `upstream/FreeToken` 保持原样；权重大文件及运行日志放在仓库外。许可及第三方署名见 [NOTICE](NOTICE)。
