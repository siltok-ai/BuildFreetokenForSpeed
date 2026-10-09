# Speed 接入约定

FreeToken 定制与素材生成由本仓库负责，Speed 的 backend、worker、服务协议和最终发布包由 Speed 仓库负责。接入是**固定并消费一套匹配的素材**，同时更新 Speed 自己的 API adapter。

机器可读摘要见 [consumer-contract.json](../integration/speed/consumer-contract.json)，模型示例见 [examples](../integration/speed/examples/)。

## 使用哪些文件，最终放到哪里

| 输入 | Speed 最终运行位置 / 行为 |
| --- | --- |
| 定制 wheel | 安装 / 暂存到 `app/python/Lib/site-packages/`，包括 `freetoken/` 与对应 `.dist-info/` |
| wheel 内 3 个 native `.pyd` | `app/python/Lib/site-packages/freetoken/kernel/` |
| `kernel-cache/<spec>/<spec>.dll` | `app/runtime/kernel-cache/<spec>/<spec>.dll` |
| cache manifest / metadata | 随 cache 一起保存，供版本与库存校验 |
| 模型 manifest | Speed 模型目录中；示例的相对权重路径需自行绑定 |

wheel 不能仅取 Python 源码、丢弃发行元数据或 `.pyd`；DLL 不能与别的版本 wheel 混用。模型权重不由本仓库提供。

## 开发环境

先在完整、独立的 SourceVenv 安装同一 wheel；配置：

```text
service.freetoken_kernel_cache_dir = <素材目录>/kernel-cache
```

使用 venv 安装时由它的 site-packages 提供 FreeToken。若用独立 runtime root，配置 `service.freetoken_runtime_dir`，并保留完整 `freetoken/` 和 `.dist-info/`；仅指向源码 `python/` 不满足版本校验合同。

运行 native / cache 自检、Speed worker 和真实模型回归后，再进入完整发布流程。禁止用 `.pth` 借另一个 venv 来满足交付闭包；Speed stager 按 SourceVenv 实际发行元数据及文件收集依赖。

## Speed 必须完成的源码适配

1. 更新被锁定的定制 FreeToken 版本、wheel SHA 和 cache 合同；完整 commit 与补丁身份也应写入发布溯源。
2. 适配新 FreeToken 配置：Speed IPC 的 `moe_backend` 投影为上游 `moe_strategy`，增加 `ple_backend`；原基线已有生成事件三元组、pending 与 `PromptAdmitted` 处理，升级时保持 reasoning / text 分流、usage、EOS、取消后复用。
3. 文本路径禁用不需要的多模态 encoder，Windows 路径使用已验证 attention 能力，不能假设 Linux FlashInfer 路径可用。
4. 旧 MoE 与 dense 模型采用各自参数。dense 不传 MoE cache 字段，使用其对应 KV 规格；不能仅凭版本升级复用旧 AOT 列表。
5. 正常 worker 清除 INVALID 环境变量。实验 worker 仅对显式 `allow_invalid_model` 条目启用，并检查原索引、保留权重 SHA、拓扑和实际资源预算。
6. Flash Next 正常条目仍要求完整 checkpoint 与真实内存预算；实验条目单独命名、单独 guard，不能全局降低完整模型预算或放松缺片检查。

实际实现留在 Speed 仓库。最终分支 `feat/freetoken-win1-flash-next` 在打包基线 `5fd331bc5257a6593d6bcb4707c5fa13cf69396b` 上只有一个提交：[808003f0efc70e7f43226e5d1e8c32b3704405cb](https://github.com/siltok-ai/siltok-speed/commit/808003f0efc70e7f43226e5d1e8c32b3704405cb)。源码说明见该提交的 `silitok_speed/qwen36/FLASH-NEXT.md`；原探索快照仍保留为历史证据，本仓库不复制应用源码 patch。

最终 review 收紧 Flash Next 为实际验证的 RadixArk NVFP4 量化布局，拒绝未认证的混合格式；实验 manifest 同时识别 `Qwen36Backend` 和 `FreeTokenQwen36Backend`，同路径普通条目独立校验。QSA 只用于 Flash Next，不允许旧模型误选。这些修复均在 Speed，本轮未追加 FreeToken 补丁。

## Windows 打包

沿用 Speed 的 `packaging/windows/build.ps1`，显式提供：

```powershell
# 示意参数，应添加到 Speed 自己的完整、已审核构建命令中。
-FreetokenWheel '<素材目录>\wheels\freetoken-0.1.3+speed.win1-cp312-cp312-win_amd64.whl'
-KernelCacheDir '<素材目录>\kernel-cache'
```

SourceVenv 必须先安装同一 wheel，因为 stager 在复制前还会导入自检；单靠暂存阶段替换 wheel 不足以纠正 SourceVenv 身份。打包应使用匹配的 BuildVenv / SourceVenv，收集真实依赖库存，并更新旧默认素材路径。

最终 Speed 会检查 wheel 的**实际 SHA**，不是只检查版本或文件名。直接复用当前制品即可匹配其锁定 SHA；跨机器重新编译若字节变化，即使源码与版本相同也会被拒绝。应先登记新的构建记录与实际 SHA、完成验收，再在 Speed 独立候选中更新 `silitok_speed/qwen36/freetoken_contract.py` 的 wheel SHA 与消费身份，用新目录重新打包。源码或补丁发生变化则必须使用新的定制修订，不能覆盖已接受的 win1。禁止重命名未知 wheel、跳过 SHA 或只改 cache 版本字符串来通过检查。

本次保持 Torch ABI 组合，不能因此推断其他后端一律需要重编译；Speed 自己修改的模块仍要进入它的正式编译流程。FreeToken wheel 负责自己的 native `.pyd`，AOT DLL 独立暂存，无需把它们编进 Speed 的业务扩展。

关闭 FreeToken C++/CUDA JIT 不等于关闭 Triton 的运行编译需求。发布闭包仍须保留相应 Triton 工具、头文件和依赖。对未覆盖的 shape，不应自动放开 JIT 掩盖 cache 缺项。

## 准入与验收边界

旧两个模型使用真实 NVFP4 权重，Flash Next 当前仅使用 INVALID-alias20，见 [验收说明](validation.md)。本次 SourceVenv 审计有 101 个发行包、31,452 个 owned 文件，0 缺失、0 editable；原 SAM3 两项约束例外维持原政策。71 项 FreeToken 构建 lock 不能替代完整 Speed 应用闭包。

已执行完整 `build.ps1 -SkipInstaller`，并使用随包编译后的 Speed、embedded Python 3.12.10、FreeToken/native/cache，在清理外部 Python/CUDA 路径后完成三个模型共 6 组 HTTP/SSE 与 backend 实际运行。208 个模块源码 SHA 与最终提交对应的 worktree 全部一致；见 [随包证据](../artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/)。这是同一本机的隔离运行环境验收，尚需安装器安装 / 运行 / 卸载、干净机器和完整正确 Flash Next 验收。
