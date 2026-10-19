# 构建当前 FreeToken Windows 材料

当前入口是 [current-release.json](../integration/speed/current-release.json)，具体profile、版本及规格读取该指针和profile。Python 和 PowerShell 构建入口使用同一默认值：冻结上游归档、所选有序补丁、3个native模块、27个AOT DLL。旧profile仍可显式选择，历史材料不覆盖。

## 环境合同

| 项目 | 当前已验证组合 |
| --- | --- |
| 平台 / ABI | Windows x64 / CPython 3.12，历史解释器3.12.13 |
| C++ | 历史完整构建VS2022 toolset14.44；本机选择性编译MSVC14.50 |
| CUDA | 历史完整构建Toolkit13.0；本机选择性编译Toolkit13.2；详见对应build-record |
| Torch | 2.11.0+cu130 |
| TVM FFI | apache-tvm-ffi0.1.13.post3 |
| Triton | triton-windows3.6.0.post26 |
| GPU | SM120；历史RTX5090 D v2 24GB；本轮RTX5070 Ti 16GB |
| 依赖 | [requirements.lock.txt](requirements.lock.txt)，71项冻结版本 |
| AOT规格 | [speed-strataq2dev4-sm120.txt](specs/speed-strataq2dev4-sm120.txt)，历史27项规格；当前具体spec读取profile |

依赖锁属于FreeToken构建闭包，不是Speed完整运行环境或离线wheelhouse。其他Python、CUDA、编译器、GPU组合需要新的profile与实际验收；不能仅改文件名沿用当前身份。

```powershell
uv venv --python 3.12.13 D:\envs\ft-current
uv pip install --python D:\envs\ft-current\Scripts\python.exe `
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' `
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python D:\envs\ft-current\Scripts\python.exe -r .\build\requirements.lock.txt
```

安装VS2022的C++ x64工具、14.44 toolset和Windows SDK。PowerShell入口通过vswhere定位工具链；CUDA_HOME应指向CUDA13 Toolkit。

## 预检与构建

在仓库根执行，先预检；预检不会创建输出目录：

```powershell
.\build\build-materials.ps1 `
  -BuildPython D:\envs\ft-current\Scripts\python.exe `
  -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' `
  -OutDir D:\builds\ft-current-001 `
  -Jobs 2 -PreflightOnly
```

通过后去掉`-PreflightOnly`。OutDir必须是新目录。复现历史版时显式传入`-Profile build/profiles/win1.json`或其他旧profile。

构建会校验全部冻结输入，在独立build-source重放所选profile指定的全部有序补丁，再按其规格生成wheel、AOT DLL及实际SHA库存。当前FT9为57组补丁、3个native模块和27项AOT；切换profile时以其catalog与spec为准。源码来自本仓库归档，不在线拉取可变main，不读取其他开发worktree。输出状态`built_requires_runtime_validation`只说明构建完成。

```text
<OutDir>/
├── build-source/             应用补丁后的源码
├── preparation-record.json  输入与补丁身份
├── logs/                    补丁、wheel、AOT构建日志
├── wheels/                  含3个native模块的wheel
├── kernel-cache/            27个AOT DLL及元数据
└── build-record.json        实际SHA、工具链和producer身份
```

失败后保留输出并换新目录重试。不拼接不明DLL，不以回退JIT掩盖缺失制品。

## 校验和交付

```powershell
python -B tools/verify_current_release.py
python -B tools/verify_materials.py D:\builds\ft-current-001
python -B tools/validate_python_revision.py --help
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

`verify_materials`核对库存；`validate_python_revision.py`可进一步将wheel Python与指定profile完整重放结果比较。仓库验证保留历史win1等价证明及后续版本各自的证据，不将旧验收转移给新候选。离线检查不能代替模型/GPU验收。

复用已验收材料时，整体使用当前指针的 `materials` 目录，安装其中wheel并显式指向配套kernel-cache。Speed同时锁定wheel实际SHA；跨机器重编译可能产生不同二进制，必须另存、验收并更新Speed消费身份，不能覆盖原材料或仅复用版本号。历史review1使用perf8相邻Python修订的显式复用路径，3个native/27个AOT不重编，验证两份源码及允许边界。

完整权重、MTP可复现转换、搬迁回执和Speed配置见 [当前交付流程](../docs/flash-next-current-delivery.md)。MTP默认关闭。Speed编译包、安装器和干净机验收仍须在Speed侧完成。

## 更新依赖或历史材料

`freeze_build_requirements.py`只用于新候选环境，不能在正式构建时自动改写已验收lock。审核新依赖、输入SHA、新profile/版本及硬件验收后再接入。旧win1构建说明保存在 [历史文档](../docs/history/build-readme-before-current-20261014.txt)。上游与补丁维护规则见 [maintenance](../docs/maintenance.md)。
