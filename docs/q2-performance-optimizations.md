# Flash Next Q2 推理优化

本次保持 FreeToken 的模型、路由、prefix cache、SSD PLE、offload LRU、CUDA Graph 和 Speed worker 架构，针对完整 48×512 Q2_0 专家的实际主要瓶颈优化。权重、激活精度及专家完整性不变。

当前材料修订为 `0.1.3+speed.q2opt1`，profile 为 `build/profiles/q2opt1.json`，消费合同见 `integration/speed/consumer-contract-q2opt1.json`。旧 `q2exp1` 素材与原始证据保留。

## 三项采纳方向

1. **大 prefill 按专家分组做矩阵计算。** 旧算法为每个 token × top-k 单独做向量计算，反复读取/解码同一专家权重。复用 FreeToken 的 GPU `moe_align_block_size`，对 ≥128-token prefill 执行 BF16 grouped GEMM，FP32 累加；Q2 解码只在寄存器 tile 内发生，无整层 BF16 展开。decode 保留原向量算法与 CUDA Graph。
2. **小残余 prefill 按需加载专家。** 当新 token ≤16 且路由数 ≤专家数的 1/3，Q2 pure GPU offload 复用原 decode LRU / gather / slot 映射，避免搬运全部专家。attention/GDN 的 prefill 状态处理不变。
3. **Q2 默认 chunk 从 1024 扩大到 4096。** 分组计算缩短后，重复的全专家银行搬运变成主项。4K 输入只遍历一次银行，可进一步降低 TTFT。此默认只由 Speed 的 Flash Next Q2 profile 设置；NVFP4 和显式小 chunk 保持原策略。

## 原理与来源

第一项将具有权重复用的向量计算转为矩阵计算；第二项按实际路由需求搬运；第三项在瓶颈变化后减少重复遍历。未使用额外激活量化、SM 专用汇编或特异性算子调参。

参考了 Strata `6674a0065fb96bacde33e3eb10f91a1df86f95f2` 的[分组 prefill](https://github.com/Niko1221/Strata/blob/6674a0065fb96bacde33e3eb10f91a1df86f95f2/src/prefill/moe_fused.cu)和 prompt-cache 短尾的 decode 思路。Strata 的 int8 激活量化、C++ runtime 和专用 CUDA 内核没有移植。FreeToken 官方源码固定于 `078132480ea4a0424f677a27eb22085624d331e6`，本地差异仍通过有序补丁维护。

## 本机完整模型 A/B

RTX 5090 D v2 / Ryzen 9900X / Windows 11 / Torch 2.11.0+cu130；同一完整混合权重、cache1024、KV8192、greedy；每长度 3 个冷请求。下表采用最后的基线复测，避免把第一轮 decode 波动算成改动收益。

| 原生输入边界 | q2exp1 TTFT | q2opt1 + chunk4096 TTFT | 降低 |
|---|---:|---:|---:|
| 冷 70 token | 0.746 s | 0.746 s | 约相同 |
| 冷 1024 token | 2.200 s | 0.753 s | 65.8% |
| 冷 4096 token | 8.753 s | 1.092 s | 87.5% |
| 冷 6144 token | 13.157 s | 1.823 s | 86.1% |
| 热 70 token，新6 | 0.751 s | 0.091 s | 87.9% |

原生 decode 仍约 48–49 token/s，未宣称本轮提升 decode。热 4K/6K 剩余64 token 不在小残余策略范围内，仍约 0.73–0.75 s。单层 GPU 微测的 8–11× 加速不是端到端加速。

原生A/B使用从冻结q2exp1 wheel解包的独立基线和候选runtime；候选仅替换三份性能源码，因此原生记录的版本标签仍为q2exp1。最终q2opt1 wheel包含完全相同的三份性能源码，并更新版本号/元数据；新wheel另经实际Speed HTTP与backend验收。原始记录保留真实标签，源码逐字节对应写入验收摘要。

完整 Speed localhost HTTP/SSE 的 4K TTFT 中位数约 1.108 s，单请求 IPC decode 约48 token/s，含末帧的 SSE 约41 token/s；未修改服务收尾策略。HTTP 全程100ms容量采样 commit 最小余量约1.77 GiB，使用既有 Flash Next 验收的1 GiB commit /2 GiB VRAM safety，严格准入与全局默认保持不变，pagefile不改。原生请求 CUDA reserved 峰值约13.18 GiB，不等同于整卡已用或整机commit。

## 数值与验收边界

权重恢复仍为 BF16，激活/中间结果 BF16，乘加/路由归并 FP32；矩阵计算改变了浮点累加顺序，因此不保证逐位一致。GPU 合成真实几何的相对 L2 差异约0.00021–0.00036，独立参考、集中路由、空专家与 padding 的7项测试通过。最终 native36组对照中34组 token完全相同，2组长输入回答措辞发生变化；算术703和完整374-token thinking 的 `FINAL:76` 均通过。不能宣称综合质量或所有上下文一致。

新 wheel 的实际 Speed backend 已验证4K输入、完整thinking、结构回答、取消后同worker复用及卸载；HTTP/SSE也通过。129项Speed定向测试及材料仓库离线源码/来源检查通过。三个模型的最终质量验收和精确源码身份、原始输出、命令与容量数据见 `artifacts/q2opt1-cp312-cu130-sm120/validation/performance-20261010/`。本轮不宣称重建完整Speed安装包、安装器或干净机器验收。

## 可重放的新材料

优化只涉及Python/Triton算法与MoE搬运调度，原生扩展、AOT源码和26规格均不变。`build/derive_python_materials.py`先重放新旧冻结源码，拒绝任何未界定的原生/AOT输入变化，再创建新版本wheel、重建RECORD、按字节复用3个.pyd和26个DLL，并记录基础材料和逐文件SHA。它不是重编译，也不覆盖原材料。新wheel SHA为 `babbc0b8d0316c2b570a9b1d3294ac51e79ae5ef8bd6f6894a537ab327370c31`。

```powershell
python -B build/derive_python_materials.py --profile build/profiles/q2opt1.json --output <新目录>
python -B tools/validate_python_revision.py --materials <新目录>
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

如需从相同冻结输入重新编译全部原生材料，仍可使用 `build/build-materials.ps1 -Profile build/profiles/q2opt1.json`。二进制SHA不同必须重新验收并更新Speed锁。

## 本轮结束范围

不直接扩大专家cache：历史约11%的decode收益需要完整服务commit余量，当前资源条件不适合默认扩容。PLE gate与gather blocks历史无明确收益，不重复探索。MTP、主干再量化和服务GC路径涉及更大范围，留给独立任务；本轮已有明确端到端收益，完成交付后结束。
