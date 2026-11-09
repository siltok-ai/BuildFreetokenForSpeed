# 在 Speed 中使用 FreeToken 材料

当前材料由 [current-release.json](../integration/speed/current-release.json) 选择。先核对 Speed 的 `silitok_speed/runtimes/freetoken/release.json`：FreeToken 版本、Windows/CPython ABI、Torch/CUDA、AOT 规格必须配套。

当前选择交付 `0.1.3+speed.cloverq8.2`，默认入口读取 release 指针，无需额外选择 profile。检出与 Speed release 合同匹配的材料修订，记录两仓完整 commit；分支名称不参与配对。

在本仓库根执行：

```powershell
$selected = Get-Content integration/speed/current-release.json -Raw | ConvertFrom-Json
$profile = Get-Content $selected.profile -Raw | ConvertFrom-Json
$materials = Join-Path (Get-Location) $selected.materials
$wheel = Join-Path $materials ('wheels/' + $profile.wheel_filename)
$cache = Join-Path $materials 'kernel-cache'
python -B tools/verify_portable_consumer.py --speed-root D:\src\siltok-speed
```

上面的检查针对仓内现成材料。使用[自行构建](../build/README.md)的产物时，将 `$materials` 改为本次输出根（含 `wheels` 和 `kernel-cache`），再计算 `$wheel`、`$cache`；按构建说明检查新产物。仓内检查通过不代表已检查这个外部目录。

版本不匹配时先同步配套的 Speed 与当前材料；不要混用不同材料目录中的 DLL。现成 wheel/AOT 已在 Git 中，消费它们无需安装 CUDA 编译工具链。

## Speed 源码或普通 wheel

先按 Speed 自己的说明准备所需运行环境，再安装本仓库 wheel，并指定外部 AOT：

```powershell
$python = 'D:\envs\speed\Scripts\python.exe'
& $python -m pip install --force-reinstall --no-deps $wheel
$env:SILITOK_FREETOKEN_KERNEL_CACHE_DIR = $cache
$env:FREETOKEN_KERNEL_CACHE_DIR = $cache
& $python -m silitok_speed.runtimes.freetoken.release self-check --pip-check --kernel-cache $cache
```

使用 uv 创建环境时可加 `--seed` 安装 pip。`--pip-check` 仅用于确实安装了 pip 的环境。
源码不变而重新编译的 wheel 可以沿用相同版本号；这里显式重装，避免 pip 因版本相同而继续使用环境中的旧二进制。
完整 Speed 环境如包含已登记的依赖约束例外，自检时按 Speed 文档附加 `--constraint-policy <Speed仓库>/packaging/windows/runtime-requirements.toml`。

FreeToken 可导入和组件自检通过，不代表 Engine 已获得设备授权。源码/CI 测试授权用法、服务启动命令及 API 授权见实际检出的 [Speed 仓库](https://github.com/siltok-ai/siltok-speed)中 `doc/speed/models/freetoken.md`，以该版本的说明为准。

## Windows 安装包

按实际检出的 Speed 仓库中 `packaging/windows/README.md` 准备应用环境并执行其构建入口，显式传入：

```powershell
# 将以下参数加入 Speed Windows 打包说明中的完整 build.ps1 命令：
# -FreetokenWheel $wheel -KernelCacheDir $cache
```

SourceVenv 中安装同一个 wheel，安装器消费配套 kernel-cache。Speed 检查版本、依赖、ABI 和文件布局，不将 wheel/AOT 固定到某个二进制 SHA。本仓库现有材料来源校验用于构建端，不增加产品运行时校验。

`FREETOKEN_DISABLE_JIT=1` 禁止原生扩展临时构建，不会关闭 Triton 的形状编译。匹配的 Triton 运行依赖仍由 Speed 发布闭包提供，不能因已有 AOT 而省略。

保护安装版使用包内 Python；其加载器不适用 `python -m` 模块入口，embedded Python 也不要求 pip。诊断应通过导入调用：

```powershell
$app = 'D:\Speed'
$env:SILITOK_FREETOKEN_KERNEL_CACHE_DIR = Join-Path $app 'runtime/kernel-cache'
& "$app\python\python.exe" -B -c 'from silitok_speed.runtimes.freetoken.release import main; raise SystemExit(main(["self-check"]))'
```

安装器升级、依赖服务处理和设备授权属于 Speed。模型资源的独立制作与挂载步骤见 [资源包说明](flash-next-gguf-bundle.md)。
## 可选函数调用部署

支持函数调用的 Speed 应使用独立部署清单：Flash 配置命令
`tools/flash_next_speed_config.py` 与 Clover 制包命令 `tools/clover_bundle.py`
均可加 `--enable-tools`，生成 `features: ["chat", "tools"]` 和
`tool_call_format: "qwen3_coder"`。默认不加该参数，保留现有聊天部署。

工具清单要求 Speed `0.2.0+freetoken.tools1` 对应改动或后续包含它的版本。
旧 Speed 不认识此格式，不能消费工具清单；升级时在新目录生成配置，保留旧包。
FreeToken 仍为 `0.1.3+speed.cloverq8.2`，wheel/AOT/权重和模型模板不变。
Speed 负责工具控制、XML 参数解析、标准 HTTP/SSE 输出及工具结果回传；
外部 Agent 负责实际执行函数。部署说明及真实客户端验证入口见配套 Speed 文档。
