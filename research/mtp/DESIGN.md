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

单请求语义状态为118,063,112字节，包括GDN recurrent、conv、全部声明slot_states（PLE ngram/conv）及QSA pending ring。原型已经实现两种事务：恢复后重放有效前缀，以及逐行checkpoint后直接提交最后有效行。后者复用原GDN任意行track机制，再正确补足窗口前的conv/PLE历史和每行QSA ring。当前诊断借用池中4个空闲slot，不增加原池容量；正式引擎必须预留专属checkpoint所有权和预算，不能在prefix压力下假定总有4个空slot。

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

## 2026-10-14 实际循环与性能

`speculative_loop.py` 已实现单请求greedy的实际草拟、causal窗口验证、部分提交、target R追齐及退出后恢复原请求。`row_snapshots.py` 提供逐行事务；`draft_graph.py` 将1/2/3草稿链和1/2/3有效行追齐放入私有CUDA graph，位置和候选传递保留在GPU，只在整链返回时读CPU。草稿注意力复用FreeToken分页FP32 GQA，GPU掩码仅允许读取当前cell及以前；SM120 FP32 KV使用num_stages=1以满足共享内存上限。主模型27个AOT、正式wheel和安装环境字节没有改变。

验证窗口的PLE不能直接用普通prefill `conv1d` 替代原decode的四tap乘积求和。旧路径在第105个输出token分歧；恢复后replay与逐行checkpoint两条路径产生完全相同的512-token序列，单独改GDN conv或逐行MMVQ也未修复。将短窗口PLE向量化为原decode的tap布局/求和，并沿用原GDN decode conv逐行计算后，512-token输出完全匹配冷基线。独立原生decode审计覆盖64个实际输出：每个验证行的GDN recurrent/conv、PLE conv/ngram和QSA ring误差均为0。审计必须与性能测试分进程；混合审计/测量的首轮进程明显扰动了verify时间，原始低速结果仍保留。

隔离审计后，8064永久专家、10112总槽的实际512-token原型测到短上下文123.12/123.00/114.86 token/s（中位123.00），首组4K上下文120.71。全部输出ID及恢复后的原生成均匹配对应无MTP冷基线。三草稿链中位约1.84ms，有效行追齐/提交约0.22–0.28ms；这些速度是同步提交后在独立循环中的实际token吞吐，**不是Speed worker或HTTP MTP性能**。prompt prime、每次新建graph及其预热单列，不包含在decode计时内。

当前仍没有正式MTP profile、版本wheel、worker多token交付或Speed MTP开关。尚须实现可复用的动态请求/slot窗口图、专属事务池及资源准入；验证EOS/长度/stop、取消、prefix复用/分叉/压力、缓存重建、thinking和非greedy回退；随后绑定独立发布材料并测真实Speed/TCP。现有原型的强制事务归还和独占页预留只适用于诊断，不能直接作为生产调度器。短上下文/4K同容量对照和更多4K重复测量继续单独保存；不能只用一次最快值声明完整MTP验收。

## 正式引擎开发 checkpoint：2026-10-14

正式开发源码独立位于 `C:/workspace/artifacts/flash-next-strata-q2-20261012/mtp-engine-development-v1/build-source`，本地分支 `develop/strata-q2-mtp-engine`。它由冻结dev7 profile重放，原始checkpoint为 `07e7213`，验证窗口与KV所有权改动为 `b4b3985`。新改动已经导出至 `patches/series/strataq2mtp1-0014-verification-foundation.patch`；对应来源、测试、失败记录和逐文件重放SHA位于 `evidence/engine-foundation-20261014-v1/index.json`。这只是开发checkpoint，尚无新发布profile/wheel，测试显式声明源码覆盖和dev7 native来源，不能继续用dev7版本字符串宣称它是原正式制品。

第一份正式差异新增独立 `verify` batch/FLA/PLE标识，因果窗口仅允许单个已有状态的请求、1–4行。GDN复用原decode卷积，GGUF递推按每行写入track槽；conv、PLE卷积与ngram快照都包含进入窗口之前的历史，解决短窗口不能只从当前输入取历史的问题。QSA ring快照读取GPU元数据中的实际请求槽，不再捕获诊断请求的固定CPU table_idx。原 `ForwardOutput` 三字段ABI保持不变；新batch上的可选token bundle尚未接入调度器。

KV的 `mtp_reserved_len` 是物理页面的排他所有权边界，不是可读语义长度。普通准备流程也记录实际申请边界，避免原流程已经申请pending token所在页之后，窗口再申请一次并覆盖页表。重复预留不会重复计费；仅 `cached_len` 之前的有效token可进入prefix tree；结束/取消沿整个已申请边界释放拒绝尾部。对齐结束时保留有效prefix并独立释放未来页，不对齐结束时释放全部未捐赠页。16组跨页/接受长度组合及共享prefix测试均通过页面总数、唯一性和树命中检查。

组件结果为34项测试与12个subtest通过，包含GDN/PLE在CUDA上的全部窗口行状态及输出与native decode完全一致、验证graph切换live slot仍一致、原混合prefix捐赠和缓存rebuild。CPU PLE向量化SiLU有约1.2e-7尾数差异，CPU输出断言使用2e-7容差；所有状态及CUDA输出仍要求精确相等。失败v1/v2和修正后v3日志完整保留。组件结果不能代替真实全模型、取消或服务吞吐验收。

### Prefix与草稿KV的所有权约定

MTP草稿cell p使用主模型R_p与token p+1；其KV依赖后一token，不能直接随位置p的共享prefix页发布，否则不同续写会互相覆盖。正式设计将cell p的KV放在**主模型位置p+1的物理token槽**。长度P的共享prefix仅携带cell 0..P-2；分叉处的cell P-1由所恢复的最后R和新token P在该请求新页中重算。保留最后主R作为声明的FP32 slot state，它随GDN/PLE一起COW、快照和重建。草稿递归产生的未验证R/KV不发布到共享页；只发布由真实target R追齐后的有效cell。这样可以复用prefix，同时避免将依赖未提交token的状态发布给后续请求。

正式运行时仍需实现和验证这项辅助分页KV、最后R持有、专属四行事务槽预算、可复用窗口/草稿graph、EOS/stop裁剪、多token调度交付、取消、工具锚点和缓存重建。普通采样在正确拒绝采样完成前走原无MTP生成，但其有效prefix的辅助状态也必须维护一致。不会将当前诊断的整段KV预留或临时全局hooks直接作为服务实现。

### 正式prefix状态与资源 checkpoint

第二份源码提交 `66ae9a2`、补丁 `patches/series/strataq2mtp1-0015-owned-prefix-state.patch` 已落实上面的所有权接口：辅助KV随主位置p+1物理页持有，分配新页时清除valid标记，prefix恢复只读取0..P-2草稿cell。主模型在最终HC混合之前捕获FP32 R，并将最后R及每个track边界R写入声明的slot state，随正常COW/reset/rebuild处理。QSA pool统一管理辅助KV与其重建，不另设页面分配器。四个事务槽是池的额外保留槽，不会被请求分配、prefix淘汰或idle reclaim交还到普通free list。

配置默认 `mtp_runtime_path=None`；关闭时不分配辅助KV、最后R捕获或额外事务槽。开启时新增成本进入模型slot声明、状态池上下限与KV cost/factory，实际量化模型配置检查得到9→13物理状态槽、最低5→9；状态池1,062,346,824→1,535,033,448字节；每64-token页成本1,671,168→1,933,376字节，QSA ring/scratch固定成本61,440→110,592字节。4096行R buffer为167,772,160字节，预先分配后才求解缓存预算。最终还须加入草稿权重/private KV/graph成本，不能用当前组件几何宣称总资源预算已验收。

65项测试与12个subtest通过：真实HybridRadix prefix分叉、GDN live状态COW、辅助KV共享/分叉、旧物理页重用失效、QSA rebuild与逐张量成本、最后R的CPU/CUDA捕获/COW/reset、专属槽所有权，以及前一checkpoint的native窗口数值/graph测试。源码与重放逐文件SHA一致，记录在 `evidence/engine-prefix-state-20261014-v1/index.json`。目前只接入状态接口，generation driver、草稿prime/publish与多token调度尚未接入，完整模型生命周期/服务验收仍未完成。

Strata实现的逐项对照和缺口见 `STRATA-MAPPING.md`。本轮将8份固定版本MTP/verifier/prefill/controller/采样核心参考保存为MIT来源锁，避免仅依赖量化工具来解释推理设计。普通greedy草稿dense注意力本来也是Strata选择；其32K/ring、受限head、异步事件和动态k需结合本机单GPUtrace选择适用实现。

### 无MTP与MTP的分支和测量隔离

无MTP Build分支 `explore/flash-next-strata-q2`、Speed分支 `explore/freetoken-flash-next-strata-q2` 及dev7制品保持不变。MTP改动分别留在 `explore/flash-next-strata-q2-mtp`、`explore/freetoken-flash-next-strata-q2-mtp`，以独立profile、版本、wheel/cache合同和默认关闭的配置接入，禁止覆盖原制品。正式实测保留原最佳无MTP、按MTP资源预算调整后的无MTP和相同预算开启MTP三条结果，并明确单独引擎/backend/HTTP口径。用户已授权完善并验证后推送远端；当前尚未推送，也没有合并main。
