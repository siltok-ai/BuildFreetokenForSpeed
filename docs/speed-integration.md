# 在 Speed 中使用 FreeToken 材料

当前材料由 [current-release.json](../integration/speed/current-release.json) 选择。先核对 Speed 的 `silitok_speed/runtimes/freetoken/release.json`：FreeToken 版本、Windows/CPython ABI、Torch/CUDA、AOT 规格必须配套。

在本仓库根执行：

```powershell
$selected = Get-Content integration/speed/current-release.json -Raw | ConvertFrom-Json
$profile = Get-Content $selected.profile -Raw | ConvertFrom-Json
$materials = Join-Path (Get-Location) $selected.materials
$wheel = Join-Path $materials ('wheels/' + $profile.wheel_filename)
$cache = Join-Path $materials 'kernel-cache'
python -B tools/verify_portable_consumer.py --speed-root D:\src\siltok-speed
```

版本不匹配时，选择与 Speed release 对应的本仓库修订，或升级为配套的 Speed；不要混用不同材料目录中的 DLL。现成 wheel/AOT 已在 Git 中，消费它们无需安装 CUDA 编译工具链。

## Speed 源码或普通 wheel

先按 Speed 自己的说明准备所需运行环境，再安装本仓库 wheel，并指定外部 AOT：

```powershell
$python = 'D:\envs\speed\Scripts\python.exe'
& $python -m pip install --no-deps $wheel
$env:SILITOK_FREETOKEN_KERNEL_CACHE_DIR = $cache
$env:FREETOKEN_KERNEL_CACHE_DIR = $cache
& $python -m silitok_speed.runtimes.freetoken.release self-check --pip-check --kernel-cache $cache
```

使用 uv 创建环境时可加 `--seed` 安装 pip。`--pip-check` 仅用于确实安装了 pip 的环境。
完整 Speed 环境如包含已登记的依赖约束例外，自检时按 Speed 文档附加 `--constraint-policy <Speed仓库>/packaging/windows/runtime-requirements.toml`。

FreeToken 可导入和组件自检通过，不代表 Engine 已获得设备授权。源码/CI 测试授权用法、服务启动命令及 API 授权由 [Speed 使用说明](https://github.com/siltok-ai/siltok-speed/blob/feat/flash-next/doc/speed/models/freetoken.md) 维护。

## Windows 安装包

按 Speed 的 [Windows 打包说明](https://github.com/siltok-ai/siltok-speed/blob/feat/flash-next/packaging/windows/README.md) 准备应用环境。在 Speed 仓库执行其构建入口，显式传入：

```powershell
# 将以下参数加入 Speed Windows 打包说明中的完整 build.ps1 命令：
# -FreetokenWheel $wheel -KernelCacheDir $cache
```

SourceVenv 中安装同一个 wheel，安装器消费配套 kernel-cache。Speed 检查版本、依赖、ABI 和文件布局，不将 wheel/AOT 固定到某个二进制 SHA。本仓库现有材料来源校验用于构建端，不增加产品运行时校验。

保护安装版使用包内 Python；其加载器不适用 `python -m` 模块入口，embedded Python 也不要求 pip。诊断应通过导入调用：

```powershell
$app = 'D:\Speed'
$env:SILITOK_FREETOKEN_KERNEL_CACHE_DIR = Join-Path $app 'runtime/kernel-cache'
& "$app\python\python.exe" -B -c 'from silitok_speed.runtimes.freetoken.release import main; raise SystemExit(main(["self-check"]))'
```

安装器升级、依赖服务处理和设备授权属于 Speed。模型资源的独立制作与挂载步骤见 [资源包说明](flash-next-gguf-bundle.md)。
