# 生成与复制 Windows 素材

当前构建输入由 [profiles/q2opt2.json](profiles/q2opt2.json) 固定，使用[六组当前补丁](../patches/q2opt2-series.json)。默认构建 / 准备入口均选择 q2opt2。以下 ft-win1 环境命名及历史制品说明保留作为工具链参考；复现历史 win1 必须显式指定 `-Profile build/profiles/win1.json`。正常构建使用冻结归档与补丁，不从可变 worktree 复制源码，也不会在线拉取最新 main。

## 本次环境合同

| 项目 | 已验证组合 |
| --- | --- |
| 平台 / ABI | Windows x64，CPython 3.12；历史解释器为 3.12.13 |
| 工具链 | VS2022 C++ toolset 14.44，`cl` 19.44.35228 |
| CUDA Toolkit | 13.0，历史 `nvcc` 13.0.48 |
| Torch | 2.11.0+cu130 |
| TVM FFI | apache-tvm-ffi 0.1.13.post3 |
| Triton | triton-windows 3.6.0.post26 |
| GPU 目标 | SM120；实际验证设备为 RTX 5090 D v2 24GB |
| Python 构建依赖 | [requirements.lock.txt](requirements.lock.txt)，71 项版本 |
| AOT 规格 | [specs/speed-sm120.txt](specs/speed-sm120.txt)，26 项 |

当前 producer 针对这一个 Windows / CUDA13 / SM120 合同实现；`-Profile` 可指定输入位置，但不意味着已经支持任意 Python、编译器或 GPU 架构。更换组合需要调整构建入口并建立新 profile、新版本和新的实际验收。

## 创建独立环境

以下使用 `uv`，也可使用等价的正常虚拟环境。路径按目标机器调整，不复用另一个 venv 的 `.pth`：

```powershell
uv venv --python 3.12.13 D:\envs\ft-win1
uv pip install --python D:\envs\ft-win1\Scripts\python.exe `
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' `
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python D:\envs\ft-win1\Scripts\python.exe `
  -r .\build\requirements.lock.txt
```

这份 lock 固定发行包版本，未镜像第三方 wheels，也未为所有第三方包附 wheel 哈希；安装仍需要对应下载源可用。完全离线交付需另行保存依赖 wheelhouse 及逐文件 SHA，不能仅凭版本 lock 声称全依赖永久离线可重建。

VS2022 需要安装 C++ x64 工具、14.44 toolset 与 Windows SDK，CUDA_HOME 指向 CUDA13 Toolkit。PowerShell 入口通过 `vswhere` 查找 VS2022 并初始化 x64 环境，优先使用指定 CUDA。

## 先预检，再构建

在仓库根目录执行：

```powershell
.\build\build-materials.ps1 `
  -BuildPython D:\envs\ft-win1\Scripts\python.exe `
  -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' `
  -OutDir D:\builds\ft-win1-001 `
  -Jobs 2 -PreflightOnly
```

预检核对输入 SHA、CPython 3.12、71 项安装版本、Torch/CUDA/TVM FFI 与 CUDA/MSVC 编译器。预检不创建输出目录。通过后去掉 `-PreflightOnly`，正式构建；`OutDir` 必须是新的目录。

编译流程：

1. 校验冻结输入与上游 commit，对独立 build-source 应用 4 组补丁。
2. 调用补丁提供的 `tools/build_windows_extensions.py bdist_wheel`，生成带 3 个 `.pyd` 的定制 wheel。
3. 调用补丁提供的 `tools/build_windows_kernel_cache.py`，用 26 项规格为 SM120 编译 AOT DLL。
4. 生成 cache 构建元数据、`speed-kernel-manifest.json` 与实际 SHA 库存。
5. 输出 `build-record.json`，状态为 `built_requires_runtime_validation`。工具成功编译不会自动宣称模型运行通过。

输出：

```text
<OutDir>/
├── build-source/            应用补丁后的可读、可调试源码
├── preparation-record.json 输入与补丁身份
├── logs/                   每组补丁、wheel、AOT 阶段日志
├── wheels/                 定制 wheel
├── kernel-cache/           26 DLL、两份 cache 元数据
└── build-record.json        实际制品 SHA、工具链和 producer SHA
```

失败时立即停止并保留输出与日志。处理后用新目录重新构建；不拼接未知来源旧 DLL，也不通过重新开启 JIT 掩盖制品缺口。

## 构建后的验收与复制

```powershell
python -B tools/verify_materials.py D:\builds\ft-win1-001
uv pip install --python D:\envs\ft-win1\Scripts\python.exe `
  D:\builds\ft-win1-001\wheels\freetoken-0.1.3+speed.win1-cp312-cp312-win_amd64.whl
```

校验只证明库存和文件身份。还需要在对应设备上完成原生加载、SSD row-store 数值 / IO 测试、旧模型 Speed HTTP/SSE/取消/复用与新模型对应验收；方法和证据见 [验收说明](../docs/validation.md)。完整发布另需 Speed 编译和目标运行包验收。

给其他机器复用已有结果时，直接复制 `artifacts/win1-cp312-cu130-sm120/`，按同一命令校验。给其他机器重新生成时，复制整个本仓库，再准备工具链和依赖；没有硬编码依赖本机原 FreeToken worktree、Speed worktree 或历史日志中的盘符。

交给 Speed 打包的素材为 `<OutDir>/wheels/<wheel>` 和 `<OutDir>/kernel-cache/`，使用 `-FreetokenWheel` / `-KernelCacheDir` 参数；SourceVenv 需要先安装同一个 wheel。详见 [接入约定](../docs/speed-integration.md)。

Speed 最终适配还锁定 wheel 的实际 SHA。同一冻结源码跨机器重编译后，二进制字节可能变化；不能仅提供相同版本文件名就替换原材料。新输出先验收、登记 SHA，再在 Speed 候选分支更新 wheel SHA 与消费身份并重做打包。直接复制本仓库已有 wheel / 26 DLL 则保持当前锁定身份。本轮完整 Speed 随包运行使用现有材料通过，没有发现 producer 或 FreeToken 补丁需要进一步修改。

## 维护依赖 lock

`freeze_build_requirements.py` 用于从准备好的 Windows 环境生成**新候选**依赖闭包，不在正式构建过程中运行：

```powershell
D:\envs\ft-candidate\Scripts\python.exe -B build/freeze_build_requirements.py `
  --output .\out\candidate-requirements.lock.txt
```

审核新 lock、创建新 profile、更新输入 SHA 并实际验证后才接入。不要从任意当前环境自动重写已验收的 win1 lock。该脚本冻结 FreeToken 构建闭包，不等于整个 Speed 应用依赖闭包。
