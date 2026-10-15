# Flash Next Strata Q2：RTX 5070 Ti 交接实测

2026-10-15，本机交接验收记录。完整 Q2 GGUF、原生 FreeToken、Speed backend 和真实 TCP HTTP/SSE 已运行；新的 Cython 整包已构建、真实安装并完成回归及卸载清理，按用户明确决定带已知问题接受；独立 rebase 整合版仍待重新验收。

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

## 探索分支实际安装验收

正式 f01 从干净提交 `e0524007` 构建，版本 `0.1.2.dev0+strataq2mtp2.2.2026101514`；安装器 SHA256 为 `b07aa4d1c66c33dc69bca0d7a62cd3e6762d182ede2dc398ff5e69af3d6db89e`。已经真正安装并使用包内 embedded Python、受保护 Cython 模块和随包 cache 测试，没有回退到源码或开发环境。

相同 HTTP 512-token 协议，三档各六次实际输出均完成：

| 模式 | 短输入 decode 中位 tps | 4K输入 decode 中位 tps |
| --- | ---: | ---: |
| MTP关闭，永久3584 | 71.63 | 68.84 |
| MTP开启，永久2304 | 61.10 | 62.62 |
| MTP关闭，永久2304 | 66.57 | 63.97 |

这组数字仅证明安装版对应性能测量完成。包内 backend 的完整20题也保持15/20及全部历史token IDs一致，thinking 3/3；正常安装服务健康、同步与SSE已运行。

**尚未解决的观测：**安装版HTTP的一次长逻辑题，在约925个正常输出token后开始持续输出`!`，直至2048输出上限，未通过原质量判据。单字符`!`对应token ID 0；失败时没有采集worker原始IDs和logits，925为文本重编码估计。源码同环境重跑、安装版原样重跑均自然结束1308tokens且回答与原源码逐字符相同；另一次带数值观察器的10轮长题全部正确，13,737次采样未见NaN/Inf。NaN/Inf只是待验证假设，尚未确认根因，也没有因此修改引擎或放宽检查。观察器可能改变运行时序，其成绩不能代替无观察器验收。

随后无数值观察器的24轮长题中，原输入12/12正确；新增前缀变体11/12正确，剩下一题重复自然语言推理而达到2048输出上限，没有重现连续`!`。这一新增失败输入在直接调用原生FreeToken时产生了与安装HTTP版逐字相同的错误回答，属于当前模型运行配置的质量边界，不能将它归为Speed打包引入的回归。原生路径的原题冷、热两次均正确且与源码回答一致。这里没有通过更改提示、采样或结束规则，把任何失败重判为成功。

用户为进一步专项排查给出30分钟预算：如无明确定位且不能稳定触发，可保留失败证据，作为已知风险继续剩余验收和分支交付。专项排查已按用户后续明确指示停止，未修改引擎或放宽质量判据。原Qwen3.6 35B backend/HTTP及H3安装回归均已完成；H3实际生成124帧832×480视频与双声道AAC，全部可解码、画面有运动、音频有限且非零。测试安装i03已正常卸载，15项清理检查通过。

## 运行档位与余量

`sm120-16g-measured` 使用上表容量：off GPU预算14.875GiB/host33GiB，on及off-matched GPU预算14.875GiB/host34GiB；GPU准入安全余量256MiB、commit安全余量4GiB。off只剩约0.49GiB显存，这是接近本机上界的实测方案，需要足够空闲显存。

`sm120-16g-roomy-measured` 保留1GiB GPU准入余量、4GiB commit余量：off永久2816/总4864、GPU预算14GiB/host34GiB，on及off-matched永久1536/总3584、GPU预算14GiB/host35GiB。off原生推理的短/4K decode 中位约69.43/66.27 tps，整轮最低空闲显存1.432GiB；该数是原生路径，不能直接写成HTTP结果。两档的Speed backend完整质量及取消复用均已通过。

较小专家池的适配留在Speed包装层：要求显式GPU与host预算，继续验证完整专家所有权下界、固定动态槽、KV和MTP约束，保留资源准入及监督保护。没有通过专家别名、部分权重、关闭资源检查或更换量化精度让模型启动。更小驻留池牺牲命中率与速度，是16GiB卡相对历史24GiB卡的主要配置差别之一；本轮未做硬件瓶颈的独立归因实验。

## 正确性与交付边界

完整非thinking质量集保持历史Q2的15/20，20题输出token IDs全部匹配历史基线；thinking三题3/3。历史5个失败没有改判或从测试中移除。原生、backend和HTTP覆盖算术、长短输入、连续请求、取消后同worker复用、卸载重载、完整进程树退出；MTP另覆盖最终提交窗口取消、后续输出与未取消参考一致，以及非greedy回退。原Qwen3.6 35B和H3的源码及实际安装版回归均已完成。

原始报告、资源采样、固定提示及源码差异位于 [本轮源码证据](../research/acceptance/flash-next-5070ti-20261015/source/README.md)。交付配置生成和显式cache用法见 [当前交付流程](flash-next-current-delivery.md)。历史5090 D v2约100–110tps的记录保留历史硬件限定，不作为5070 Ti的通过门槛。

本页当前不证明长期压力运行、并发模型、完整8192边界、更长上下文、27B dense在16GiB的运行资格、全新操作系统或完全离线安装。整包和整合分支通过后，应分别追加其提交、安装器SHA、实测结果与清理事实，保留本轮源码数据。

探索消费合同：[consumer-contract-5070ti-explore.json](../integration/speed/consumer-contract-5070ti-explore.json)。实际安装封存149文件见[explore](../research/acceptance/flash-next-5070ti-20261015/explore/README.md)。`python -B tools/verify_current_release.py`会复算安装身份、性能、质量和清理；Speed用`python -B tools/verify_flash_next_checkpoint.py --build-root <Build仓库>`检查当前代码与合同。旧`verify_mtp_checkpoint.py`保留历史哈希，仅用于历史提交，不篡改为新验收。署名：boqi。
