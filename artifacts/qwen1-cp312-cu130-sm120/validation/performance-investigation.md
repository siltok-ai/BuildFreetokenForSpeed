# Qwen3.8-27B 性能对照

测试硬件：RTX 5090 D v2 24 GB、Windows、CUDA 13.0、Torch 2.11.0+cu130。
固定官方 FreeToken 078132480ea4a0424f677a27eb22085624d331e6，最小 Windows 材料 0.1.3+speed.qwen1。
独立脚本直接调用官方 LLM，不导入 Speed；只在输出回调记录事件时间，不修改推理和采样内核。

| 相同输入 45 token | 直接 FreeToken | Speed 源码 |
|---|---:|---:|
| 输出 128 token，三次中位数 | 57.54 token/s | 57.62 token/s |
| 输出 512 token，三次中位数 | 57.31 token/s | 57.07 token/s |

输入为同一个英文技术说明 prompt，temperature=0、top_k=1、thinking=false、ignore_eos=false。
每次都实际生成到请求上限。先 warmup 16 token，decode=(输出事件数-1)/(末事件时间-首事件时间)，不含 prefill。
测试顺序为 FreeToken 后 Speed，未随机交叉，且没有置信区间；不到 0.5% 的差异不构成适配降速证据。

独立环境：C:\workspace\envs\speed-qwen013-dev（此任务新建）。缓存与脚本在本任务 artifacts 中。
没有使用既有任务工作区或修改其他 venv。独立脚本及完整输出为 direct-freetoken-bench.py、direct-freetoken-qwen38.json；
Speed 对照为 speed-comparison-bench.py、speed-comparison-qwen38.json。

官方 state_dict() 中 1302 个张量全部在 cuda:0，张量共 20,161,558,532 字节，没有 CPU 权重。
实际量化方法：129 个 Nvfp4LinearMethod、128 个 Fp8TensorLinearMethod、48 个 UnquantizedLinearMethod。
CUDA 已分配 21,963,307,008 字节，reserved 22,489,858,048 字节。KV 为 4096 token，CUDA graph 最大 batch=1。
这份 NVFP4 checkpoint 是混合精度；跳过视觉和 MTP，运行的是全 GPU 驻留的 dense 文本模型。
固定官方快照没有该模型的投机解码执行路径，未开启 MTP、DFlash 或 DSpark。

公开材料（2026-10-13 查阅）：

- [Murai Labs 原始 5090 测试](https://github.com/Murai-Labs/qwen3.8-27b-rtx5090)：单张普通 5090、unsloth NVFP4、vLLM；256/2048 输入、200 输出、并发1，普通解码 51.5/52.5，MTP K3 为 107.6/97.3 token/s。权重与引擎均不同，只作性能量级参考。
- [FreeToken 原始 3090 测试](https://github.com/OthmanB/localLLMinference/blob/main/research/qwen3.8-27b-context-runtime-benchmark-2026-08-30.md)：同一 RadixArk checkpoint/revision，FreeToken 0.1.2、3090 限功耗250W、8K/16K/32K 输入约27.5–31.5 token/s，硬件、功耗、上下文不同，不能外推为5090 D v2指标。
- [官方 FreeToken 路线图](https://github.com/FlashML-org/FreeToken/issues/79)：投机解码包含 MTP/DFlash/DSpark 的规划；本次判断同时核查固定快照代码，不能将 fork 的功能视为官方支持。
- [NVIDIA 5090 D v2 规格](https://www.nvidia.cn/geforce/graphics-cards/50-series/rtx-5090-d-v2/)：24 GB、1344 GB/s；与32 GB普通5090不同。

此次未找到 FreeToken + Qwen3.8-27B + 5090 D v2 的同条件公开性能结果。
结论：57 token/s 是本机固定 FreeToken 路径的可重复实测值，Speed 没有表现出明显额外开销。
不能据此证明已达到硬件极限；更高的公开投机解码或多并发数字不适合直接对比。
