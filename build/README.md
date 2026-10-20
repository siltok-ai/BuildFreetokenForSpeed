# 构建 FreeToken Windows wheel 和 AOT

当前 [release](../integration/speed/current-release.json) 选择 profile、固定上游归档、58 个有序补丁及 AOT 规格。完整构建生成 3 个原生模块和 27 项 AOT DLL；直接消费现成材料见 [Speed 接入](../docs/speed-integration.md)。

## 准备环境

| 项目 | 当前构建要求 |
| --- | --- |
| 系统 / Python | Windows x64 / CPython 3.12 x64，开发头文件和导入库完整 |
| C++ | VS2022 C++ Build Tools、MSVC 14.44、Windows SDK |
| CUDA Toolkit | 13.0，包含 nvcc、头文件和开发库 |
| GPU 目标 | SM120 |
| 依赖 | [requirements.lock.txt](requirements.lock.txt)：FreeToken 构建环境的固定版本 |

Toolkit 是实际编译器；Torch 的 cu130 标签是运行时版本。完整 Speed 安装包的其他依赖由打包者自行准备。

在仓库根执行。以下示例使用 uv，先装官方 Torch，再安装其余固定依赖，避免多个索引的同名包选择问题：

```powershell
uv venv --seed --python 3.12 D:\envs\ft-build
$python = 'D:\envs\ft-build\Scripts\python.exe'
uv pip install --python $python --no-deps --index-url https://download.pytorch.org/whl/cu130 `
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130'
uv pip install --python $python -r build/requirements.lock.txt
```

## 完整构建

```powershell
.\build\build-materials.ps1 `
  -BuildPython $python `
  -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' `
  -OutDir D:\builds\ft-001 -Jobs 2 -PreflightOnly
```

预检通过后去掉 `-PreflightOnly`。预检不创建输出目录；正式构建要求 OutDir 尚不存在。入口通过 vswhere 寻找 VS2022，并显式选择 14.44 toolset。

```text
<OutDir>/
  build-source/             从归档解压并重放补丁的源码
  preparation-record.json  本次输入和补丁目录
  logs/                    wheel / AOT 编译日志
  wheels/                  包含 3 个原生模块的 wheel
  kernel-cache/            27 个 DLL 和缓存元数据
  build-record.json        实际工具链与产物库存
```

```powershell
& $python -B tools/verify_materials.py D:\builds\ft-001
& $python -B tools/validate_python_revision.py --materials D:\builds\ft-001
```

源码检查默认使用当前 profile。更换 profile 时同时传 `--profile`。新产物状态为 `built_requires_runtime_validation`；安装该 wheel、显式使用新 kernel-cache 后完成 GPU 组件及 Speed OFF/ON 推理检查，再用于交付。重编译二进制可与仓内版本不同，Speed 不固定其 SHA。

## 使用另一套工具链

保持当前 profile 不变，在仓库外复制它并增加 `build_toolchain`，例如：

```json
"build_toolchain": {"cuda": "13.2", "msvc": "19.50"}
```

这里指定编译器版本；profile 中 `torch`、`cuda` 仍描述运行时依赖。通过 `-Profile D:\cfg\ft-local.json -VsRoot D:\tools\vs2026 -ToolsetVersion 14.50` 明确选择另一套编译工具，再执行相同预检与完整构建。入口仍检查实际编译器，不自动接受版本不匹配。使用新目录保存输出，并在目标 GPU 验证；不能仅凭构建成功推断兼容性。

## 仅 Python 修订重新封装

当前 portable1 的原生代码和 AOT 输入与 FT9 相同，可从保留的基线重新封装：

```powershell
& $python -B build/derive_python_materials.py `
  --profile build/profiles/strataq2portable1.json `
  --base-profile build/profiles/strataq2perf9.json `
  --base-materials artifacts/strataq2perf9-cp312-cu130-sm120 `
  --output D:\builds\ft-python-001
```

工具重放两份源码、验证允许的差异并复制原生材料，重新生成 wheel RECORD。原生代码、公共头文件、AOT 或 ABI 改变时应完整构建；选择性原生重编入口只供明确维护相应算子的开发者使用。

修改构建依赖时在独立环境更新 lock 和 profile，按 [维护规则](../docs/maintenance.md) 验证；`freeze_build_requirements.py` 不在正式构建中自动更新锁文件。
