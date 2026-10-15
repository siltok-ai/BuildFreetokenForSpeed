# Flash Next Strata Q2：RTX 5070 Ti 交接实测

2026-10-15，本机源码阶段记录。完整 Q2 GGUF、原生 FreeToken、Speed backend 和真实 TCP HTTP/SSE 已运行；新的 Cython 整包、实际安装版以及 rebase 整合分支仍待验收。本页随各阶段补充，当前数字不能代表安装包已经通过。

## 环境与配置

Windows 11、Ryzen 7 9700X、RTX 5070 Ti 16GB（NVML 实际总量约 15.92GiB）、64GB RAM，驱动 616.56。系统分页文件维持 3968MiB，commit 上限约 65.49GiB，未为验收更改系统设置。测试串行运行单个 GPU worker，避免其他 GPU 负载。

FreeToken 为冻结的 `0.1.3+speed.strataq2mtp2`，使用原 wheel 和配套 27 个 AOT DLL；没有修改引擎、量化计算或权重。Python 3.12.10、Torch 2.11.0+cu130、Triton Windows 3.6.0.post26、TVM FFI 0.1.13.post3。Speed 的本机适配提交为 `e0524007b5d27f1cdef28ffb5fa60fc45d36d5ed`，从历史 `9791fc7c` 加入小驻留池显式预算适配。原始性能采样时的源码差异另外封存；正式安装版和最终整合提交将分别记录。

模型为固定 revision 的完整两片 GGUF（共 66,423,878,624 字节），MTP 使用固定来源的 31 份预测层张量。各模式均为 KV8192、动态专家槽1024、prefill 专家槽1024、disk PLE、radix、graph batch1、prefill chunk4096。逻辑接口支持的最大上下文不等于本轮实测范围；本轮长输入为4096 tokens，输出512 tokens。

## 真实 HTTP/SSE 性能

使用 Speed 的 `tools/bench/benchmark_freetoken_http.py`，经真实 TCP 接收 SSE。短输入70 tokens、长输入4096 tokens，各连续运行3次；每次均实际交付512 tokens，保留 EOS，`finish_reason=length`、usage、512个内容事件及唯一 DONE 均已核对。测试前有16-token预热；首次使用提示与两次立即重复都计入下表中位数，不重置专家或前缀缓存。

| 模式 | 永久专家 / 总槽 | 短输入 decode 中位 tps | 4K输入 decode 中位 tps | 整轮最低空闲显存 GiB | 整轮最低可用 commit GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| MTP关闭，已测最快配置 | 3584 / 5632 | 71.76 | 68.94 | 0.487 | 9.387 |
| MTP开启 | 2304 / 4352 | 61.04 | 62.64 | 0.588 | 7.527 |
| MTP关闭，同驻留容量对照 | 2304 / 4352 | 66.56 | 63.37 | 2.133 | 9.359 |

这里的 tps 为 `(实际输出token数-1)/(最后内容事件时间-首内容事件时间)`，不含 TTFT，不是请求总耗时吞吐。显存与 commit 由独立监督进程每0.5秒采样，属于整轮观测值，不能作为瞬时最低值或其他负载下的保证。NVML 的 used 包含驱动保留部分，不能直接与 nvidia-smi 的 used 混算；此处统一报告 NVML free。

在这些输入和容量下，MTP没有加速：同容量短输入约慢8.3%，4K输入约慢1.1%；与各自已测最快档相比差距更大。因此5070 Ti继续默认关闭MTP。六组同容量开/关输出文本、usage及结束原因完全一致。这里没有宣称所有提示、未来配置或其他设备的MTP都无收益。

短输入 TTFT 实测约0.79–1.00秒；4K首次使用约12.50–12.86秒，立即重复约1.49–1.53秒。这是已有任务缓存下的观测，包含前缀缓存状态等影响，不是空缓存首启保证或单独的 prefill 算子吞吐。首次遇到新的形状时还可能触发 Triton JIT；应另外保留冷编译、模型加载和 TTFT 记录，不能把它们藏进稳态 decode 数字。

## 运行档位与余量

`sm120-16g-measured` 使用上表容量：off GPU预算14.875GiB/host33GiB，on及off-matched GPU预算14.875GiB/host34GiB；GPU准入安全余量256MiB、commit安全余量4GiB。off只剩约0.49GiB显存，这是接近本机上界的实测方案，需要足够空闲显存。

`sm120-16g-roomy-measured` 保留1GiB GPU准入余量、4GiB commit余量：off永久2816/总4864、GPU预算14GiB/host34GiB，on及off-matched永久1536/总3584、GPU预算14GiB/host35GiB。off原生推理的短/4K decode 中位约69.43/66.27 tps，整轮最低空闲显存1.432GiB；该数是原生路径，不能直接写成HTTP结果。两档的Speed backend完整质量及取消复用均已通过。

较小专家池的适配留在Speed包装层：要求显式GPU与host预算，继续验证完整专家所有权下界、固定动态槽、KV和MTP约束，保留资源准入及监督保护。没有通过专家别名、部分权重、关闭资源检查或更换量化精度让模型启动。更小驻留池牺牲命中率与速度，是16GiB卡相对历史24GiB卡的主要配置差别之一；本轮未做硬件瓶颈的独立归因实验。

## 正确性与交付边界

完整非thinking质量集保持历史Q2的15/20，20题输出token IDs全部匹配历史基线；thinking三题3/3。历史5个失败没有改判或从测试中移除。原生、backend和HTTP覆盖算术、长短输入、连续请求、取消后同worker复用、卸载重载、完整进程树退出；MTP另覆盖最终提交窗口取消、后续输出与未取消参考一致，以及非greedy回退。原Qwen3.6 35B和H3的源码回归已经运行，新安装版仍需重复。

原始报告、资源采样、固定提示及源码差异位于 [本轮源码证据](../research/acceptance/flash-next-5070ti-20261015/source/README.md)。交付配置生成和显式cache用法见 [当前交付流程](flash-next-current-delivery.md)。历史5090 D v2约100–110tps的记录保留历史硬件限定，不作为5070 Ti的通过门槛。

本页当前不证明长期压力运行、并发模型、完整8192边界、更长上下文、27B dense在16GiB的运行资格、全新操作系统或完全离线安装。整包和整合分支通过后，应分别追加其提交、安装器SHA、实测结果与清理事实，保留本轮源码数据。
