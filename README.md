# FreeToken 0.1.3：Speed dev 双模型材料

此分支只维护 Qwen3.6-35B-A3B 与 Qwen3.8-27B dense 文本运行需要的 Windows 材料。官方完整源码固定为 `078132480ea4a0424f677a27eb22085624d331e6`，源码版本 0.1.3；它不是 v0.1.3 tag。定制版本为 `0.1.3+speed.qwen1`。

## 输入与补丁

`upstream/` 保留官方原貌和归档 SHA。`patches/qwen1-series.json` 是唯一补丁目录，`build/profiles/qwen1.json` 是唯一构建 profile。

1. Windows 运行：`.dll` AOT 加载，串行 safetensors 读取时跳过不可用的 POSIX page-cache hint。
2. Windows 构建：CUDA/MSVC 参数、四个 AOT 源文件的编译兼容、固定规格构建工具。
3. 包身份：独立本地版本号。

相对官方源码只修改 12 个文件。不包含我们为 Flash Next 增加的 RowStore/SSD PLE、Q2 专家、专家别名、Q2 prefill 优化或权重转换工具。官方快照自身的架构代码保持完整。

历史候选的归档、补丁和材料保留在 main 与 Git 历史；本分支不覆盖其字节或版本。

## 固定工具链

Windows x64 / CPython 3.12 / Torch 2.11.0+cu130 / CUDA Toolkit 13.0 / MSVC 14.44 / apache-tvm-ffi 0.1.13.post3 / triton-windows 3.6.0.post26 / SM120。

`build/requirements.lock.txt` 固定 FreeToken 构建依赖版本，不代表整个 Speed 产品闭包，也不保证第三方下载永久可用。

## 构建与复用

```powershell
python -B tools/prepare_source.py --output D:\builds\qwen-inspect
.\build\build-materials.ps1 -BuildPython D:\envs\qwen\Scripts\python.exe -CudaHome D:\CUDA13 -OutDir D:\builds\qwen-001 -PreflightOnly
# 预检后去掉 -PreflightOnly；输出必须是新目录。
python -B tools/verify_materials.py artifacts/qwen1-cp312-cu130-sm120
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

输出为一个 wheel（`_pinned_tensor` 与 `_cpu_moe` 两个官方扩展）和 19 个同次构建的 AOT DLL。dense 必需的 `index_10240` 与 `store_2048` 已包含，规格不包含七个 Flash Next 专用项。所有 DLL 与 wheel 均记录实际 SHA；重新编译字节变化必须更新消费合同并重新验收。

## Speed 消费

`integration/speed/consumer-contract.json` 绑定 wheel、cache、来源和验收范围。Speed adapter 放在 Speed 仓库：更新版本和材料锁、纯文本关闭编码器、`moe_strategy` 映射、Windows Qwen MoE 串行读取。

交付前必须执行真实生成、thinking、HTTP/SSE、取消后复用与卸载，并在 dev 正式编译运行目录中重复。离线校验成功不代表 GPU 或随包验收成功。

状态与实际性能见 `artifacts/qwen1-cp312-cu130-sm120/validation/`。本机没有设备证书；用户授权的许可证临时绕过仅在外部验收进程中存在，未写入 Speed 正式源码或材料。安装器和干净机器验收另行记录。
