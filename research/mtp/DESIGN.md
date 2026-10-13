# FreeToken 上的 Flash Next MTP

本分支从已验证无MTP版本分叉：Build `42b79d2096036b65485d800eece5de4980da7930`（引擎/材料 `ab06451`），Speed `a0fc86d9a8278c1c815893b319d25ae952b0ce4e`（运行代码 `23311dbd`）。原分支、checkpoint、wheel和27个AOT保持不变。这里的来源工具和窗口原型不代表已实现MTP服务。主体继续是Speed的FreeToken worker；Strata仅作格式、数学和流程参考。

## 模型与权重

主模型继续绑定完整ISTA-DASLab GSQ-RCO两片GGUF的固定revision。预测层从Qwen BF16固定revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` 提取31个 `mtp.*` 张量，来自28个safetensors分片，总计5,214,301,696字节。只发HTTP范围请求，必须206及精确Content-Range，不下载360GB主检查点，不回退main。当前下载已完成，每个原始文件分别通过Strata冻结参考SHA及完整集合核验。权重独立放在 `C:/workspace/models/Qwen/Qwen3.8-Flash-Next/<revision>/mtp-source-bf16`，与GSQ和RadixArk区分。

`fetch_source.py` 使用本目录MIT参考及其字节锁；`build_runtime.py` 逐专家转换，在新目录生成FreeToken原始Q2_0布局。沿用Strata预测层的17候选scale MSE选择，FP16 scale和相邻四个2-bit码，解码 `(code-1)*scale`，gate半块之后是up半块。**它与Strata S2的gate/up交织和scale/code分平面布局不同**，禁止直接把S2当FreeToken bank。512个草稿专家都驻GPU。较大的预测投影Q8_0，HC/router BF16配FP32激活，norm为原始zero-centered FP32，运行时只加一次1。每份派生数据保存来源、量化方法、几何、字节数、SHA及独立重构样本；新来源不覆盖已验收资料。

## 预测计算与目标验证

预测层接收主模型最终HC混合**之前**的四流残差 `R[T,10240]` 和下一token embedding。hidden norm对全部10240元素归一化，fc_hidden共享2560×2560权重逐stream投影；embedding norm/fc后广播加到四流。随后一层QSA、HC、512/top10 MoE及shared expert，再经最终HC mixer和head。主embedding/head优先借用现有压缩数据；是否建立受限词表head以实测接受率决定。草稿自身的K/V独立；dense注意力在≤2051个cell时与QSA等价，超过阈值仅改变草稿接受率，不改变经验证的主模型输出。

位置遵循Strata cell约定：主模型位置p的R配位置p+1的token，**草稿RoPE位置仍为p**，预测p+2。连续草稿以草稿自身R递归，不把尚未存在的主模型R当输入。主prefill要为草稿生成正确K/V；必要时为这项接口提供连续R副本，所有算子仍须支持实际合法stride，不要求全模型输出强制连续。

验证窗口是同一请求的连续2–4个输入token，使用causal-prefill元数据，GDN只更新一次该请求的槽。不能把decode batch4图当成一个请求的4个token，这会破坏递推顺序。增加独立静态窗口graph、全部窗口行的target logits和R、动态位置/长度/页表/输入staging；保留原单token decode图。PLE在重放前按完整CPU历史stage多行数据，并正确填graph pinned-buffer、WAIT flag及consumed event；当前普通prefill的host ctx不能直接代替这套graph提交。

一轮输入 `[t0,d1,d2,d3]`，target各行输出 `[v1,v2,v3,v4]`。t0是已经交付但还未前推的token；从d1==v1开始计算最长匹配前缀a。交付匹配草稿 `d1..da` 和修正/bonus `v(a+1)`，最后者成为下一轮尚未前推的token。主状态只提交t0与匹配草稿的a+1行。草稿K/V按有效target R和下一token追齐，拒绝尾部不可见并随后覆盖。

## 状态事务、取消与采样

验证前备份该请求的GDN recurrent、conv和全部声明slot_states（包括PLE ngram/conv）以及QSA pending ring。当前实际单请求快照为118,063,112字节，而非此前粗估75MB。全部接受可直接保留最终状态；部分接受先恢复原地址，再以正确输入重放已接受前缀，暂用这条完整可解释的正确实现。后续若需避免replay，应设计逐行状态checkpoint并实测总收益。

分页KV及压缩index页按committed长度可见，未接受尾部不能发布到prefix tree；pending ring属于跨调用状态，必须回退或按有效前缀重建。专家LRU/驻留命中统计是性能缓存，可以保留预测验证中已搬运的专家；不能把它和语义状态混淆。取消在窗口提交边界处理，IPC只发送已验证token，prefix/GDN捐赠也只能来自已提交状态。异常恢复需使用同一事务，不能留下未验证token、悬挂PLE WAIT或槽所有权泄漏。

首个原型明确限定单请求greedy；非greedy请求先走原无MTP路径，保持已有sampling语义。要支持temperature/top-p/penalty，需实现与target一致的拒绝采样或耦合策略，不能直接以argmax匹配宣称任意采样正确。EOS、max_tokens、stop和cancel按实际交付长度裁剪；最后尚未前推的token延续无MTP行为。

## 资源与验收

当前最佳无MTP11264槽余量只有数百MiB，不能直接额外加载约0.9GB预测层。启动先计入512个草稿专家、投影/head、草稿KV、R handoff、118MB事务快照和窗口graph/scratch，再确定主专家永久驻留数。8064永久+1024动态+1024prefill（10112总槽）作为容量探索候选，精确释放1,592,524,800字节主专家GPU bank；它尚不是新的服务默认值。Speed manifest/ready握手必须绑定新的几何、真实host/GPU字节和MTP身份，不能沿用旧预算。

性能至少比较：原9216驻留无MTP、按MTP预算缩小驻留的无MTP、同预算MTP。固定历史提示、温度、512个实际交付token、专家/prefix冷热口径，多轮记录最终交付tps、草稿长度/接受数、draft/verify/replay时间、H2D/命中率、TTFT及NVML/commit峰值。目标是最终交付更快；草稿生成速度不代表收益。继续用trace检查主要开销，不把微小尾数调查放在优先级前面。

正确性覆盖全部接受、首/中/末位置拒绝、每种提交长度、未对齐QSA group和跨page、长上下文稀疏选择、prefill切块、冷热prefix/分叉、EOS/长度/stop、取消复用、缓存resize/reset和关闭worker。独立引擎通过后再生成新profile/补丁系列/版本wheel，完成Speed backend、真实TCP和安装包验证。原始无MTP对照不得替换，尚未通过的场景必须明示。

## 已完成的独立原型

`evidence/20261013-v1/index.json` 保存来源、转换、窗口与草稿校准的完整日志/脚本、失败原因及SHA。目标窗口v2/v4在实际位置70、72上，连续四步与四行causal验证的logits和所有检查状态均完全相同；强制拒绝后提交1/2/3行并续算正确，恢复调度器后正常输出不变。v1的虚拟CPU历史不完整、v3的conv wrapper在capture中 `.item()` 失败均保留；没有改写安装环境或正式dev7。

v4用已知single-request窗口长度调用原有varlen conv静态参数，绕过host同步；专用graph同时正确提交多行PLE。位置72的暖重放中位约7.74/9.51/11.19/13.01 ms（1/2/3/4行），logits和GDN/conv/PLE/ring误差都是0。此时位置固定，尚须动态staging；这不是MTP最终速度。

预测层完成FreeToken格式转换，总823,887,872字节；实际加载含8192-cell独立FP32 KV，所拥有GPU tensor总857,442,304字节，main R捕获另167,772,160字节。初次32-token、29个完整三草稿窗口的真实FreeToken算子校准，最长匹配前缀≥1/2/3的比例为82.76%/72.41%/58.62%，以每轮真实target R追齐草稿。暖eager三草稿链约5.4ms，prefill仅K/V构建约848ms；这些只是诊断数据，未减少target forward次数，也没有宣称交付速度提升。主模型32-token输出与无MTP基线完全相同。下一步实现动态窗口及实际speculative循环后，才能测最终性能。
