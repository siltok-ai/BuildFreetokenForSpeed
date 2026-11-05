# 构建 FreeToken Windows wheel 和 AOT

当前 [release](../integration/speed/current-release.json) 选择 Clover Q8 profile、固定上游归档、66个有序补丁和34项AOT规格。前65个历史补丁保持原样，第66个补丁整合Clover运行时及预编译GGUF扩展。完整构建生成4个原生模块；不能用旧材料重新封装替代。直接消费材料见 [Speed 接入](../docs/speed-integration.md)。

## 环境

Windows x64、CPython 3.12 x64（含开发头文件和导入库）、VS2022/MSVC 14.44、CUDA Toolkit 13.2，目标SM120。依赖以 [requirements.lock.txt](requirements.lock.txt) 和当前profile为准。Toolkit是编译器；Torch仍使用2.11.0+cu130/CUDA13.0运行时。完整Speed应用的其他依赖由打包者维护。

```powershell
uv venv --seed --python 3.12 D:\envs\ft-build
$python = 'D:\envs\ft-build\Scripts\python.exe'
uv pip install --python $python --no-deps --index-url https://download.pytorch.org/whl/cu130 `
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130'
uv pip install --python $python -r build/requirements.lock.txt
```

## 完整构建

在仓库根执行，输出根必须尚不存在：

```powershell
.\build\build-materials.ps1 -BuildPython $python `
  -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2' `
  -OutDir D:\builds\ft-001 -Jobs 2 -PreflightOnly
```

预检通过后去掉 `-PreflightOnly`。入口通过vswhere寻找VS2022，选择14.44工具集，并检查当前profile声明的实际编译器。输出包括重放源码、独立wheel/AOT日志、四模块wheel、34个DLL及库存与SHA256记录。原生模块为 `_pinned_tensor`、`_cpu_moe`、`_row_store`、`_gguf`。

```powershell
& $python -B tools/verify_materials.py D:\builds\ft-001
& $python -B tools/validate_python_revision.py --materials D:\builds\ft-001
& $python -B tools/verify_clover_candidate.py --speed-root D:\src\siltok-speed --materials D:\builds\ft-001
```

新产物标记 `built_requires_runtime_validation`。库存检查不执行推理；安装同一wheel、使用配套cache后，在目标GPU冷启动三个Clover模式并回归Flash Next MTP OFF/ON及Qwen。运行时保持 `FREETOKEN_DISABLE_JIT=1`，GGUF使用wheel内原生模块；Triton形状编译仍需包内运行依赖及可写缓存。Speed打包入口复制完整依赖闭包，并验证embedded Python及实际worker。

## 工具链与历史材料

当前profile将编译器声明与运行时ABI分开。使用另一套编译器时复制profile，修改 `build_toolchain`，通过 `-Profile`、`-VsRoot`、`-ToolsetVersion` 明确选择，在新输出目录完整构建和验收，不覆盖既有材料。

历史mtpdecode2是仅Python修订，允许按其显式profile、base_profile及原有材料重新封装；它的复用资格不适用于Clover。新引擎源码涉及原生扩展、公共头或AOT时必须完整构建。当前消费者校验同时支持完整构建与历史复用合同，且拒绝将复用记录冒充完整构建。

设备许可、5090调优及安装器部署按实际范围另行验收；不从源码和库存检查推断通过。
