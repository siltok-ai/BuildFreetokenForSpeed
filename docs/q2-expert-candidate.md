# Flash Next Q2_0 专家候选

此候选保留完整 48×512 路由专家，使用原生 Q2_0 字节常驻 DRAM；HF BF16 主干、router、共享专家、tokenizer 及 SSD FP8 PLE 保持原有语义。仅消费 GGUF 第一片的 144 个专家张量，不将它声明为完整 GGUF 模型，不读取其主干或第二片 PLE。

当前版本为 `0.1.3+speed.q2exp1`，固定上游 `078132480ea4a0424f677a27eb22085624d331e6`。候选 profile 是 `build/profiles/q2exp1.json`；win1 的 profile、历史补丁和制品保持原字节。候选不包含 INVALID expert alias 补丁。

专家块是 64 个元素 / 18 字节：FP16 scale 和 16 个相邻 2-bit 编码字节，编码 0..3 对应 -1、0、1、2。GPU 内核在读取时恢复并舍入为 BF16 权重，使用 BF16 激活和中间结果、FP32 乘加及路由加权求和。它当前支持 TP=1、纯 GPU offload、SiLU、输出侧路由加权；不支持 CPU/hybrid 或 resident 策略。主干的量化配置独立处理，不能把 Q2_0 误标为 NVFP4。

专家来源为 `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 固定 revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` 的 `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`，37,623,740,192 字节，SHA256 `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`。镜像下载仍核验此固定内容身份。

`tools/prepare_q2_checkpoint.py` 重新核验原始非专家 HF 文件，然后在新目录建立硬链接和过滤后的非专家索引。普通 HF 目录使用 `--trunk-receipt integration/speed/examples/flash-next-hf-source-receipt.json`；该清单保存本次核验的原索引及 14 个非专家分片的 SHA/尺寸，原始目录仍须提供 tokenizer/config 和原完整索引。省略清单时兼容读取历史目录的下载回执，但不读取别名专家，不复制 INVALID 标记。`mixed-source.json` 记录非专家来源与校验结果；下载完成后使用 `--finalize` 验证完整专家目录和来源，填写校验时间身份。运行时拒绝不完整或更改过的来源。

已验证：

- 真实 GGUF 专家样本与独立 CPU 解码计算的数值对照，单/多 token、cache slot 重排、CUDA Graph。
- 同一专家及 gate/up/down 投影与独立 NVFP4 来源的对应关系抽查；此项只证明样本映射。
- 48 个真实 MoE 构造器仅将路由专家绑定为 Q2_0，router 和共享专家保留 BF16。
- 完整压缩 bank 的常驻、锁页及 H2D 容量探测：31.640625 GiB，RSS 增量约 31.73 GiB。
- 真实 HF 主干/SSD PLE 与人工 Q2 bank 的完整引擎容量探测：RSS 峰值约 35.49 GiB，CUDA reserved 约 11.97 GiB；该探测不用于正确性声明。

2026-10-10 已完成实际全量权重的 Speed 源码运行验收，见 [完整报告](../artifacts/q2exp1-cp312-cu130-sm120/validation/speed-http-q2.json) 和同目录的来源、数值与构建身份记录。第一片 GGUF 完整 SHA256 校验通过；运行回执确认 48×512 专家、33,973,862,400 字节（31.640625 GiB）host bank 全部锁页，无 INVALID 标记、无 alias。GPU 专家 cache 1024 槽、1.318359375 GiB，KV 8192 token。

真实推理 worker 的 RSS 为 33.605 GiB，进程历史 RSS 峰值 35.458 GiB；这两个值包含引擎运行占用，不能与专家字节混为一项。相较原完整 NVFP4 bank 核算约 63.46 GiB，Q2_0 专家字节约减半；本机未做完整 NVFP4 模型的实测 RSS A/B。

Speed 同步 HTTP 和 SSE 均返回 200，文本连贯，usage 与结束事件正常；确定性问题分别返回 `17×23 → 391`、`中国首都 → 北京`、`cat 中文 → 猫`。退出后两个 worker PID 均已消失。这证明首轮有意义的正确输出，不能推导为全面质量或参考模型 logits 一致性。首请求含冷加载约 48.90 秒，全验收约 62.66 秒；TestClient 会缓冲 SSE，报告中的 TPOT 不能用作生成速度基准。

Windows 的 commit 与 DRAM/RSS 必须分别记录。容量探测中整机 commit 峰值约 63.01 GiB，而本机 commit limit 约 65.53 GiB。Speed Q2_0 默认预算经后续实测收敛为 14 GiB VRAM + 36 GiB host，严格 commit 准入仍启用；本机默认 8 GiB commit safety 可能拒绝测试，必须明确记录测试使用的 safety 与实测峰值，不应改动系统 pagefile 来掩盖预算差异。

此次真实 Speed 验收显式设置 commit safety 为 1 GiB，严格 commit 准入通过，预测额外余量仅约 0.390 GiB；加载后状态采样整机 commit 可用约 4.434 GiB。该真实运行未连续采样整机 commit 峰值，不能把前述容量探测峰值声明为此次峰值。默认 safety 不变，系统 pagefile 不变；复现时需要足够的当前 commit 余量。

复现入口：

```powershell
python -B tools/download_q2_experts.py --output <专家源目录>
python -B tools/prepare_q2_checkpoint.py --trunk <原始 HF 文件目录> --experts <专家源目录> --output <新模型目录>
python -B tools/prepare_q2_checkpoint.py --trunk <原始 HF 文件目录> --experts <专家源目录> --output <新模型目录> --finalize
./build/build-materials.ps1 -BuildPython <独立 python> -CudaHome <CUDA13目录> -OutDir <新制品目录> -Profile ./build/profiles/q2exp1.json
```

Speed 分支应同步锁定新 wheel 的实际 SHA256；AOT cache 必须来自对应冻结构建输入。Q2_0 数值内核使用 Triton，专家拷贝默认使用已有 fused multi-bank AOT 内核。离线材料验证仍覆盖 win1 的历史等价性，候选重放与硬件证据分别保存。

本机实现保存在三个独立 worktree：Speed `C:/workspace/worktrees/speed-flash-next-q2`、材料 `C:/workspace/worktrees/build-freetoken-flash-next-q2`、引擎 `C:/workspace/worktrees/freetoken-flash-next-q2`。交付分支分别为 Speed `feat/freetoken-flash-next-q2` 和本材料仓库 `feat/flash-next-q2-experts`；不向官方 FreeToken 推送。新 wheel 和 matching cache 纳入 `artifacts/q2exp1-cp312-cu130-sm120`，wheel SHA256 为 `6dc83a35b14a2126669805e652cfb59619f4454826f5c5fc020a670b91fd8a1d`。旧分支和已验收制品保持原样。

使用已建立的独立环境再次验收实际模型（不需要重新下载）：

```powershell
& 'C:/workspace/envs/speed-flash-next-q2/Scripts/python.exe' -u -B `
  'C:/workspace/worktrees/speed-flash-next-q2/tools/smoke/freetoken_qwen36.py' `
  --model 'C:/workspace/models/qwen3.8-flash-next.bf16-q2-experts' `
  --profile flash-next --quant q2_0 `
  --kernel-cache 'C:/workspace/artifacts/freetoken-flash-next-q2exp1-build/kernel-cache' `
  --startup-timeout 600 --commit-safety-gib 1 `
  --output 'C:/workspace/artifacts/flash-next-q2-implementation/speed-http-q2-rerun.json'
```

脚本创建临时模型 manifest，通过 Speed discovery、Engine 和公开 chat API 执行验收后清理；实际混合 checkpoint 与独立环境保留，可继续开发。此命令验证源码；随包验证使用下述独立入口。

## 清理后的能力与旧模型实际回归

Speed 已删除部分专家执行入口、实验参数、索引裁剪、缺片放行和测试开关；仅保留旧 INVALID checkpoint 拒绝与继承环境清理。新引擎源码不含 alias 补丁，旧 win1 材料只用于历史复现。

下列报告均来自当前清理后的源码，使用真实完整模型。五人排序题具有位置排除、先后、相邻、间隔五项约束，独立穷举 120 种排列的唯一解为 `戊乙丁甲丙`。最终提示要求推导并检查全部约束，不限制回答字数；输出预算 2048 token。三个模型均完整答对，`finish_reason=stop`，并通过三项基础题与 HTTP/SSE、worker 退出检查：

- [Flash Next HTTP 最终验收](../artifacts/q2exp1-cp312-cu130-sm120/validation/regression-after-hack-removal/flash-next-http-final.json)：逻辑回答 1495 token；实际 worker RSS 33.609 GiB、加载峰值 35.459 GiB。
- [Qwen3.6-35B-A3B HTTP 最终回归](../artifacts/q2exp1-cp312-cu130-sm120/validation/regression-after-hack-removal/qwen36-http-final.json)。
- [Qwen3.8-27B HTTP 最终回归](../artifacts/q2exp1-cp312-cu130-sm120/validation/regression-after-hack-removal/qwen38-http-final.json)。

Flash Next HTTP 运行以 100 ms 间隔连续采样：整机 commit 峰值 63.505 GiB、最小余量 2.020 GiB，物理可用内存最小 15.816 GiB。该结果支持 host 36 GiB 的候选预算，并不意味着可以忽略 commit。测试使用 1 GiB commit safety、2 GiB VRAM safety；Qwen3.6 使用默认 8/2 GiB，Qwen3.8-27B 使用历史验收的 8/0.75 GiB。全局默认仍为 8/2 GiB，系统 pagefile 保持 3.875 GiB。更大 cache/KV 的非默认几何要求显式预算。

另有 [Flash Next 4K backend](../artifacts/q2exp1-cp312-cu130-sm120/validation/regression-after-hack-removal/flash-next-backend-long.json)、[Qwen3.6 4K backend](../artifacts/q2exp1-cp312-cu130-sm120/validation/regression-after-hack-removal/qwen36-backend-long.json) 和 [Qwen3.8-27B backend](../artifacts/q2exp1-cp312-cu130-sm120/validation/regression-after-hack-removal/qwen38-backend.json)。包含长输入、算术、thinking、五段结构化长回答、取消后同 worker 复用及卸载清理；thinking 题答案为 `FINAL: 76`，最终分别实际生成 374、2045、307 token 并正常结束。4K 输入与 thinking 是不同请求，不能据此声明同时 4K 输入 + 4K 实际输出或 256K 上下文已验证。

早期失败报告保留：短输出预算使 Qwen3.6 推理被截断；限制解释字数时 Qwen3.8-27B 给过错误排序；默认 safety 在本机也发生过加载前拒绝。最终通过结果对应去除解释字数限制、合理 token 预算与明确资源配置，不抹去这些反例，也不证明全面模型质量或参考 logits 一致性。

Speed 定向回归最终为 153 项通过，覆盖 checkpoint 完整性、Q2 来源、backend/worker、材料自检、chat reasoning/SSE 和 discovery；删除的旧实验测试已替换为实际坏目录/坏来源拒绝测试。引擎 Q2_0 单元验证 3 项通过。材料 validator 与 5 项工具边界测试仍只表示 win1 历史来源及随包证据保持有效；新候选以冻结输入重放、制品校验和本页真实模型证据为准。

## Windows 随包验证入口

独立完整产品依赖环境只替换 FreeToken 为固定 q2exp1 wheel。运行 `packaging/windows/build.ps1 -SkipInstaller` 后，输出 staged application。使用包内 embedded Python、编译后的 Speed `.pyd`、随包 wheel/native/cache，清除外部 CUDA/Python/FreeToken 环境与 PATH，仅保留 app 和 Windows 系统路径。

```powershell
python -B tools/verify_packaged_speed.py `
  --app <新构建目录>/app --speed-source <当前Speed源码> `
  --flash-next-model <完整Q2混合模型> `
  --qwen36-model <Qwen3.6-NVFP4> --qwen38-model <Qwen3.8-27B-NVFP4> `
  --output <新的验收目录>
```

工具逐模型运行 HTTP/逻辑题和 backend/长输入/thinking/取消卸载，保存完整回答、实际导入路径和编译模块 source SHA256。原始构建 manifest 记录构建时的旧基线 HEAD 与 dirty 状态，不能冒称发布 commit；编译源码哈希用于对应整理后的单提交文件树。安装器安装/卸载与干净机器验收不在本轮范围。

2026-10-10 六组随包真实验收全部通过，见 [汇总](../artifacts/q2exp1-cp312-cu130-sm120/validation/packaged-speed-20261010/packaged-models-summary.json) 与同目录完整回答、导入身份、源码证明和包装记录。Flash Next/Qwen3.6 长输入分别 4096 token，Qwen3.8-27B 为 1025 token；逻辑回答分别实际生成 1495/1557/1284 token，thinking 题分别 374/2045/307 token，均正常结束并正确。三个模型取消后 worker 复用和完整卸载均通过。

208 个编译模块 source SHA 与本机源码匹配。Windows 工作树为 CRLF，发布 Git blob 为 LF；另外保存 CRLF→LF 规范化后与已暂存发布文件树逐项比较的证明，保持原始构建身份不变。重新应用冻结 profile 后的 442 个 FreeToken Python 文件与实际 wheel 全部匹配，26 DLL 的字节身份也核验通过。

六组包内运行合计约 449 秒，100 ms 整机采样 commit 峰值 63.150 GiB、最小余量 2.376 GiB、物理可用内存最小 15.923 GiB。该数字覆盖整组 serial 验收，不能单独归因于某一次请求。新运行包本机路径为 `C:/workspace/build/flash-next-q2-review-001/app`；它与模型文件均不提交到 Git，仓库保存可重建材料和实际验收证据。
