# Strata Flash Next MTP 对照

对照固定Strata提交 `6674a0065fb96bacde33e3eb10f91a1df86f95f2`，不是通用speculative decoding示例。量化工具来源锁见 `references/origins.json`；八份引擎、验证、prefill和采样/控制器参考见 `references/mtp_engine_origins.json`，每份同时保存SHA256与Git blob身份。参考沿用MIT许可证；Speed仍执行FreeToken，不加载Strata运行时。

| Strata设计/接口 | FreeToken当前对应 | 仍需完成或评估 |
| --- | --- | --- |
| `core/mtp.hpp`：最终HC混合前R、全10240 hidden norm、逐stream共享fc_hidden、embedding分支 | 正式模型及drafter已经加载草稿、捕获R，并由正式调度读取最后R | 独立wheel、权重来源与Speed消费合同绑定 |
| cell p配R_p和token p+1、RoPE仍为p、预测p+2 | 独立链与target追齐已用该约定，六组512-token输出匹配冷基线 | 服务窗口同样保持位置与提交约定 |
| Q8_0大投影、512/top10草稿专家全驻GPU、shared expert | 已完成固定BF16来源及FreeToken原始Q2专家派生、压缩算子和真实GPU运行 | 新profile/wheel/消费合同核验来源和资源；不混淆Strata S2与FreeToken布局 |
| 草稿注意力使用dense KV，2051 cell以内等价于QSA选择，之后仅影响草稿质量 | 当前独立路径同样dense，graph复用FreeToken分页GQA | 大于当前8192容量时的ring/窗口；不能误称Strata草稿在2051之后必须做主模型QSA indexer |
| `prefill`、`prepare_prefill`：仅生成草稿prompt K/V，可复用图 | 正式prefill/普通采样回退维护辅助KV；1/2/3行catchup graph由引擎持有 | 继续在实际Speed测试中记录prime成本及TTFT |
| `Verifier.run`/`commit`：原生decode算术、有效前缀事务、QSA key tail恢复、PLE正确历史 | 正式verify标识、GDN/PLE/conv/ngram/QSA ring逐行快照；组件CUDA输出与状态精确一致 | 全模型验证及EOS/stop/取消/工具锚点；原ForwardOutput ABI不改 |
| `draft`：验证后用真实target R追齐，再递归draft | 正式调度按已提交长度推进CPU历史、GPU token池和统计，原DetokenizeMsg逐token发送 | Speed worker/HTTP实际交付与计数验收 |
| 会话恢复 `kv_restore`，有效状态才可复用 | 正式辅助KV随主位置p+1物理页持有；最后R随GDN/PLE COW；页面重用时失效标记 | 全模型prefix压力、分叉、取消和cache rebuild；这是FreeToken分页生命周期适配，不是直接复制Strata会话缓存 |
| 受限draft head、`draft_vocab.bin`，可选Q4草稿投影/head | 目前借用完整主head；已提取固定版本CJK词表并核验身份 | 按用户收敛要求，稳定版本后再A/B词表head；不能将仅英文词表作为中文默认 |
| 主模型窗口、draft/catchup私有CUDA graph；异步commit/事件顺序 | 正式引擎持有可复用target/draft/catchup图，动态更新live slot、位置及页表，事件覆盖bundle交付 | Speed实测及trace；当前MTP会先drain再准备下一窗口，无MTP保留原overlap |
| `spec/controller`按接受率及实测成本决定k，收益不够可选k=0 | 当前固定三草稿 | 运行时记录接受率/窗口成本，评估1/2/3或回退；不抄Strata旧硬件的默认成本常数 |
| `coupled_draft`/`spec_prob`：与target sampling chain一致的耦合或拒绝采样 | 当前只实现greedy，非greedy保留原无MTP采样 | 不能将argmax匹配宣称为任意采样正确；后续需要时单独验证分布/seed/penalty |

正式调度补丁0016的四组512-token已经逐项匹配冻结native冷/热基线；0017进一步限制诊断记录大小与KV lookahead容量，83项组件测试通过。41次开发引擎生成验证长度/EOS/stop、普通采样回退、取消复用、prefix分叉、重建和分块prefill。旧八条人工“pressure”提示提前EOS且没有证明保留prefix淘汰，已明确标为重复回收覆盖；实际淘汰、工具锚点及服务验收继续单列。新候选profile `strataq2mtp1`/版本 `0.1.3+speed.strataq2mtp1` 已生成独立wheel，不继承dev7的运行验收。

Strata部分 `--pipeline-windows` 机制明确面向两GPU层切分，本机单GPU不直接照搬该调度。这里优先参考适用的可复用graph、异步commit与事件顺序。当前独立MTP约123 token/s不等于上述所有能力已移植，也不等于Speed HTTP MTP性能。

正式MTP新配置默认 `mtp_runtime_path=None`：关闭时不分配草稿辅助KV、最后R捕获和专属事务槽。开启时资源分别计入物理KV成本、slot state、四个额外事务槽及预先分配的残差buffer；最终还要计入草稿权重/private KV/graph工作区。无MTP主分支及制品保持冻结，同一MTP分支也必须能明确测关闭配置，不能靠注释掉代码来制造对照。
