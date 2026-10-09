# 已有验证与验收边界

本页分别保留初轮源码验证和最终随包验收。原始文件见 [validation/](../artifacts/win1-cp312-cu130-sm120/validation/)，初轮身份见 [build-record](../artifacts/win1-cp312-cu130-sm120/build-record.json) 和 [validation-summary](../artifacts/win1-cp312-cu130-sm120/validation-summary.json)，最终结果见 [packaged-speed-20261009](../artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/)。

本机：Windows，RTX 5090 D v2 24GB，SM120，物理内存约 61.65 GiB；源码/构建 Python 3.12.13 / Torch 2.11.0+cu130。下面初轮结果来自 Speed 源码 worker；后续已验收随包 embedded Python 3.12.10，但没有生成安装器或换到干净机器。

## 两个历史模型

| 模型 | 实际结果 | 证据 |
| --- | --- | --- |
| Qwen3.6-35B-A3B NVFP4 | HTTP 同步 / SSE、usage / EOS、非法参数 400、取消后同 worker 复用、进程退出通过 | `qwen36-http-smoke.json`、`qwen36-backend-validation.json` |
| Qwen3.8-27B NVFP4 | 同上 | `qwen38-http-smoke.json`、`qwen38-backend-validation.json` |
| 两模型算术 | 均输出 703 | backend 报告 |
| Qwen3.8 thinking | 307 输出 token，正常 stop，最终 `FINAL: 76` | backend 报告 |
| Qwen3.6 thinking | 2048 上限时 1899 输出 token，正常 stop，最终 `FINAL: 76` | `qwen36-thinking-2048-validation.json` |

Qwen3.6 的 512 / 1024 上限样例尚在 thinking 时 length 截断，reasoning 有效，不是空回复或升级后的生成中断。

HTTP smoke 总时长约 42.45 / 24.63 秒，包含加载、请求与卸载，不是解码速率。FastAPI TestClient 缓冲 SSE，消费事件间隔不用于计算真实 TPS。原模型 32-token 短请求单次约 157.6 / 56.8 token/s，仅是诊断样本，未做稳定上游 / 定制 A/B。

## Flash Next：INVALID-alias20

20 个物理专家层服务 48 个逻辑层，约 26.44 GiB 唯一专家，SSD PLE；明确 `allow_invalid_model` 和保留层数开关。权重目录当时保留 30 层，不表示全部 30 层已分配进 DRAM。

| 指标 | 实测 |
| --- | --- |
| Speed HTTP / SSE / usage / terminal / 非法字段 | 通过 |
| 取消与同 worker 复用 | 通过，取消收尾约 63.5ms |
| 进程树 RSS 峰值 | 31.05 GiB |
| 设备显存峰值 | 15.55 GiB |
| 最低可用 RAM / commit | 20.85 / 5.36 GiB |
| 70 输入、128 输出 token 两轮 decode | 35.85 / 35.71 token/s |
| 上述两轮 TTFT | 2.137 / 1.417 秒 |
| 1025 输入、16 输出 token 两轮 TTFT | 3.757 / 2.833 秒 |

证据：`flash-next-alias20-http-smoke.json`、`flash-next-alias20-backend-validation.json`、`flash-next-alias20-resource-monitor.json`。HTTP smoke 总时长 140.27 秒还含首次保留权重 SHA 校验，不能归为推理速度。

输出有重复乱码，属于故意破坏专家组合后的无效输出。上述性能只验证“减少 DRAM + Windows SSD PLE + 完整执行链路可运行”，不能替代原模型正确性或与 Strata 正确模型直接比较。

## 完整模型的内存估算

完整 NVFP4 专家库约 63.457 GiB，综合加载估算进程 RSS 约 67.3 GiB；另需考虑系统、Speed、临时缓冲、GPU 映射和 commit 余量。96GB 适合受控验证，128GB 更适合完整模型且 PLE 在 SSD 的正常使用。估算不等于完整模型已在上述配置实测；本机未完成正常完整 Flash Next 推理。

## 代码与依赖验证

初轮 Speed 定向回归 124 项通过。扩展 discovery 有 6 项失败：旧视频 safetensors 夹具与 backend 隔离断言；同解释器下在原打包基线复跑得到相同失败，原始日志保留。没有将其记为通过，也没有扩大本次改动范围去修视频路径。

完整独立 SourceVenv 的 stager 审计：101 个发行包、31,452 个 owned 文件，0 missing、0 editable。自检、pip check、native 模块、Torch/TVM/Triton 与 26 项缓存库存已验证。

本仓库另提供轻量可重复检查：官方源码一致、拆分补丁等价、wheel Python 源码匹配、二进制 SHA、缓存合同和准备工具失败边界。这些可在无 GPU 的 CI 执行，不能代替上述模型实测。

本次目录整理已实际执行上述仓库校验与 5 项工具测试、独立输出源码准备、上游 archive 重导出及新版入口编译预检，均通过；重导出的归档 SHA 与历史冻结归档相同。提交后另建干净 Git clone，再次执行仓库校验、5 项工具测试与源码准备，证明 Git checkout 后冻结文件 SHA 保持不变。迁移证明见 [repository-validation.json](repository-validation.json)。这次未重复 native 编译或模型运行，旧 wheel / DLL 的来源通过逐文件等价和 hash 校验证明。GitHub CI 已配置，是否运行通过以 Actions 结果为准。

## 尚未完成

- 正常、完整专家权重的 Flash Next 正确模型运行与性能。
- 安装器产出及安装 / 运行 / 卸载。
- 干净机器安装验收；同一本机清理外部 Python/CUDA 路径的随包运行已通过。
- 长时间、多请求、稳定性能 A/B 与完整生产回归。

下一版材料应分别登记 `built`、`native_validated`、`speed_source_validated`、`packaged_validated` 等实际阶段证据，不能从工具编译成功自动推断所有阶段通过。

## 最终 Speed 打包与随包三模型验收（2026-10-09）

最终 Speed 提交 [808003f0efc70e7f43226e5d1e8c32b3704405cb](https://github.com/siltok-ai/siltok-speed/commit/808003f0efc70e7f43226e5d1e8c32b3704405cb)，分支 `feat/freetoken-win1-flash-next`，相对原打包基线只有一个提交。其使用材料 commit `12bdf743bbf4725fc25468204313cb2f43733758` 的现有 win1 wheel / 26 DLL，本轮没有重建或追加 FreeToken 补丁。

199 项定向单元 / 合同 / 打包回归通过。完整 Windows `build.ps1 -SkipInstaller` 重新编译 208 个 Speed 模块，重建并 GPU 验证 H3、收集应用依赖、验证 FreeToken native/cache、输出 runtime manifest 和 requirements lock，产物零 `.pyc`。没有生成或安装 Inno 安装器。

控制脚本使用随包 Python `-I -B`，清理外部 Python/CUDA/INVALID 环境变量，PATH 只保留 app Python 和 Windows 系统目录。工具执行前预载并检查包内编译后的 Speed、FreeToken / `_row_store`，避免工具的源码路径插入造成 Speed 源码回退。六组身份记录均保留了编译 manifest；208 个源码 SHA 与最终待提交 worktree 全部相符。

| 模型 | 两组实际运行 | 两轮32输出token诊断TPS中位数 | 取消收尾 |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B NVFP4 | HTTP/SSE 与 backend 全部通过 | 164.32 | 30.25ms |
| Qwen3.8-27B NVFP4 | HTTP/SSE 与 backend 全部通过 | 57.85 | 50.82ms |
| Flash Next NVFP4 INVALID-alias20 | 执行链路通过，语义无效 | 38.19 | 63.34ms |

三个模型均完成同步/SSE、usage/EOS、非法字段400、短/长请求、取消后同worker复用与进程退出。旧两模型输出正常；本轮没有重做此前的全部算术/thinking样例。Flash Next仍是20物理/48逻辑专家层与SSD PLE，仅用于执行与性能验证。短样本TPS不能当作稳定性能A/B或完整模型性能；本轮没有重新监测RSS峰值，不把此前31.05GiB数据记为本次打包实测。

按用户要求，打包发生在最终提交之前，所以 build manifest 真实记录原基线 commit 加 `dirty=true`。逐模块 SHA、最终 commit/tree 和材料 SHA共同绑定证据，不改写成提交后重建。报告、命令、六组运行与身份、打包/测试日志及文件SHA索引见 [封存说明](../artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/README.md)。目录中的本机路径是历史证据，不是新机器的构建输入。
