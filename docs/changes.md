# 我们对 FreeToken 的改动

基线和官方原代码见 [upstream](../upstream/README.md)；完整差异顺序、SHA 和文件表见 [series.json](../patches/series.json)。这里说明原因与边界，不把官方已有的架构 / SSD PLE / NVFP4 记作本地新增能力。

## FTWIN-001：Windows 运行与 SSD PLE

[补丁](../patches/series/0001-windows-runtime.patch)，6 个文件。

| 代码位置 | 本地工作 |
| --- | --- |
| `python/freetoken/kernel/csrc/row_store/row_store_ext.cpp` | Win32 文件句柄、UTF-8 路径转 Unicode、OVERLAPPED 定位读、对齐缓冲、NO_BUFFERING、Windows 驱动符号加载与同步原语 |
| `python/freetoken/kernel/utils.py` | Windows DLL / 加载路径兼容，使用预编译缓存 |
| `python/freetoken/models/loader.py` | Windows 加载流程兼容 |
| `python/freetoken/moe/host_banks.py` | Windows 主机专家内存实现兼容 |
| `tests/kernels/test_row_store.py` | 对应 row-store 平台验证 |
| `tests/models/qwen4_exp/test_ple_disk.py` | 对应 PLE 磁盘读取验证 |

官方已有 RowStore / PLE extent、hash、stage / flush 和 GPU 同步的架构。我们的关键贡献是把实际文件读取等平台接缝实现到 Windows，保留 SSD 数据源，而不是将整个约 47.68 GiB PLE 表搬进 DRAM。运行时仍有 staging 缓冲和其他主机状态，SSD PLE 不表示整个模型零 RAM。

已做真实 SSD native IO 路径验证，并在 Speed 实际 worker 使用 `ple_backend=disk`。上游若补齐 Windows 支持，应优先替换这些本地接缝并删除对应补丁片段，重新验证读数、对齐、取消与资源释放。

## FTWIN-002：Windows 原生扩展与 AOT

[补丁](../patches/series/0002-windows-build.patch)，9 个文件。

| 代码位置 | 本地工作 |
| --- | --- |
| `setup.py` | Windows 原生扩展构建兼容 |
| `kernel/csrc/jit/batch_memcpy.cuh`、`fast_index_copy.cuh`、`index.cu`、`store.cu` | MSVC / CUDA 编译兼容 |
| `tools/build_windows_extensions.py` | 构建 wheel 中的 native 扩展 |
| `tools/build_windows_kernel_cache.py` | 选择规格、Windows TVM 编译 flags、生成 DLL 与元数据 |
| `tools/windows-compat/sys/cdefs.h`、`README.md` | MSVC 缺少 POSIX `__always_inline` 头时的局部 shim |

这些工具进入**应用补丁后的 FreeToken 源码树**；仓库外层 `build/` 负责冻结输入、调度它们及生成 Speed 消费合同。FreeToken 内的 `tools/` 与本仓库外层 `tools/` 各有用途，不应混淆。

本次旧 dense 模型真实加载才暴露缺少 embedding `index_10240` 与 KV `store_2048` 的 DLL。现已将它们纳入 26 项规格与实际库存；仅 import / 编译成功无法验证所有模型的 shape 覆盖。

## FTEXP-001：无效专家复用，确实减少 DRAM

[补丁](../patches/series/0003-invalid-expert-experiment.patch)，13 个文件。核心位置：

- `models/nvfp4_banks.py`：仅迭代保留的专家层，跳过其余专家权重读取。
- `moe/expert_banks.py`：只调用 `alloc_layer_banks(..., keep_layers)`；其余逻辑层引用保留层张量，alpha 尺度也复制对应层，GPU cache 保留逻辑层标识。
- 新 `moe/experimental_alias.py`：双开关与范围约束。
- 新 `tests/moe/test_experimental_alias.py`：实际分配、别名、guard 与填充验证。
- 新的 9 个 `tools/*.py`：本次下载、checkpoint / 内存 / SSD / AOT 探测与 benchmark 辅助；逐项文件表在 series.json。

减少发生在**分配与读取之前**，没有先分配完整专家库再把名字改成别名。因此 unique pinned-bank 容量确实减少。48 逻辑专家层只持有 20 物理层时，专家库约 26.44 GiB；Speed 实测进程树 RSS 峰值约 31.05 GiB。

直接 FreeToken 开关为：

```text
FREETOKEN_ALLOW_INVALID_MODEL=1
FREETOKEN_INVALID_EXPERT_KEEP_LAYERS=20
```

Speed 对外使用 `engine.extra.allow_invalid_model=true` 和 `invalid_expert_keep_layers=20`，仅为明确实验条目注入环境。正常 worker 必须清除继承的两个 INVALID 环境变量。

仅用于 `qwen4_exp` 的原始 NVFP4、uniform pinned offload；不支持 FTW、layer residency、dummy 或任意量化格式。原索引与保留分片 SHA 仍需校验，实验不能变成普通 checkpoint 缺失文件自动放行。

**这是为了验证容量、Windows SSD n-gram 和推理流程而故意破坏模型的实验。** 不保证结果正确，不是完整 Flash Next 的低内存部署方案。当前 win1 wheel 内含这些代码，但默认关闭；未来正式 profile 应去掉实验差异、采用新版本号并重建，不覆盖 win1。

## FTPKG-001：定制版本身份

[补丁](../patches/series/0004-speed-wheel-version.patch)，修改 `python/freetoken/version.py` 为 `0.1.3+speed.win1`。

它将官方基线版本与 Speed 定制修订区分开来。版本只是可读标签，真正身份还包括完整上游 commit、补丁 SHA、Python ABI、Torch/TVM/CUDA/SM 和二进制 SHA。不能看到都是 0.1.3 就混用某个官方 wheel 和本仓库 DLL。

## 哪些修改留在 Speed

Speed backend 的 FreeToken API 适配、worker 事件与取消、manifest / 预算 / INVALID guard、Windows smoke junction、构建参数和 cache 库存校验属于 Speed 自身代码，继续放在 Speed 仓库。本仓库提供 [接入约定](speed-integration.md) 和模板，不把这部分私有应用代码当作 FreeToken patch 发布。

最终 Speed 适配为 [808003f0](https://github.com/siltok-ai/siltok-speed/tree/808003f0efc70e7f43226e5d1e8c32b3704405cb)，材料身份与随包验收保存在 [最终记录](../artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/record.json)；原始探索基线及适配快照 SHA 仍保存在历史 [validation-summary.json](../artifacts/win1-cp312-cu130-sm120/validation-summary.json)。本仓库提供的 FreeToken 素材不能让未经适配的任意旧 Speed commit 自动获得新 API 支持。
