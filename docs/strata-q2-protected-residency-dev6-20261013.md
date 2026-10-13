# FreeToken Strata Q2 驻留版本 dev6

正式安装的 `0.1.3+speed.strataq2dev6` 已完成独立 FreeToken 验证：无 MTP，短输入 decode 中位数 **111.99 tps**，4K 输入 **103.49 tps**。各六次测量、每次实际输出 512 token；本次最低值分别为 104.73 / 101.90 tps。这些结果适用于本机固定配置，尚不代表 Speed 或打包验收。

版本材料位于 `artifacts/strataq2dev6-cp312-cu130-sm120`，profile 为 `build/profiles/strataq2dev6.json`，新增补丁 `FTSTRATA-006`。wheel SHA256：`fa0b57ae017917608af1d2e7f02dcc006de51fbd1864200a4981990dc585c450`。三个原生模块和 27 个 AOT 内核逐字节复用已完整编译的 dev4；这是有来源证明的 Python/Triton 修订，不冒充重新原生编译。原有 profile、补丁、wheel、缓存和环境均保留。

## 实现与使用边界

专家权重仍是 FreeToken 原生 Q2 压缩 bank，使用既有 Q2×Q8 DP4A 算术。永久 GPU 专家直接加载到最终池，完全没有重复的 RAM 权重；其他专家只有紧凑锁页 RAM 所有者，可以生成可丢弃的 GPU 动态副本。prefill 双缓冲、动态 LRU 和永久驻留三个区域互不覆盖。搬运按每个专家的实际所有者指针访问，保留 Windows mapped-host 地址转换和 CUDA graph 重放。

采用冻结 Strata commit `6674a0065fb96bacde33e3eb10f91a1df86f95f2` 的官方专家排名。原始 Git blob 为 `a4ed7e5feb68e68fa76df66b9109e99c1ef143b8`，SHA256 为 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`。打入 wheel 的 JSON 排名与完整原始 STRP 数据逐项一致，没有使用测试请求训练排名。

功能显式启用，其他模型继续使用原有缓存：

```python
moe_strategy="offload"
moe_q2_resident_experts=9216
moe_q2_dynamic_slots=1024
moe_cache_size=11264
moe_cache_auto=False
max_running_req=1
num_token_override=8192
max_extend_tokens=4096
cuda_graph_max_bs=1
ple_backend="disk"
memory_ratio=0.9
```

11264 槽包括 1024 prefill、1024 动态和 9216 永久专家。锁页专家权重为 21,233,664,000 字节，GPU bank 为 15,571,353,600 字节。永久专家池在一次加载期间固定；改变容量必须重新加载模型。非法 resize 在 graph 销毁之前拒绝；KV-only 重建和相同专家容量的重建保留所有者地址。

初次完整重建测试保留了 `memory_ratio=0.8`，既有预算检查安全拒绝了 KV 重建：显式配置的专家池超过该比例的缓存预算。v2 使用 0.9 后，实际重建、graph 重捕获与后续推理通过。保留 v1 原始失败日志；不得把它计入通过结果，也不修改引擎预算算法来绕过拒绝。

## 已完成验证

- 安装后的 456 个 Python、JSON 和原生文件与冻结 wheel / 补丁重放一致。
- 23 项 GPU 测试通过：混合所有者、完全 GPU 所有的层、1/3/8-token 路由、重复路由、prefill、graph 重放、reset、拒绝 resize、保持地址的 rebuild、多实例隔离及既有投影测试。
- 完整模型通过算术、前缀分支恢复、2721-token 实际长输入、取消复用和 KV-only rebuild。662 个状态组件差异为零，12 个 QSA 选择一致；固定输出和 logits 与 dev5 对照通过。
- 20 项冻结质量题仍为 15/20；全部输出 token 与 dev5 一致，没有新增回归。无 thinking 下既有算术和格式错误保留在原始结果中。
- 12 次长输出测试的完整 token 序列与对应 dev5 / 驻留原型均一致。decode 按实际交付间隔计算；EOS 和达到 512 上限明确记录。冷缓存只清理前缀和动态副本，永久所有者保留。
- 性能测试最小 Windows commit 余量为 6,902,259,712 字节，Torch reserved 峰值约 21.80 GiB。该数值不能直接当作含 CUDA 上下文的完整显存预算。
- producer 的 18 项单元测试通过，旧版本身份校验继续通过。

验证目录 `validation/protected-residency-20261013/index.json` 保存 61 个文件及 2 个大文件外部哈希引用；外部路径位于本机研究目录，不能删除。index SHA256：`56a10b4c60c41c2dbc0dec0f62cf1bf7c22fae75dcddc2c1c88dcc7ffb16da48`。

## 接下来

停止本版驻留策略的继续探索。把此固定版本接入 Speed 候选分支，核对完整两片 GGUF 来源、IPC 配置、实际所有权回执、资源准入、HTTP/SSE、取消、退出与旧模型行为，保存可以回退的本地提交。main 不合并，远端不推送。

随后创建独立 MTP 候选，检查草拟权重、主干接口和验证路径，处理 GDN、卷积、KV、PLE 的提交与回滚、前缀复用和取消。性能必须报告真实最终输出吞吐、草拟接受率及失败回退成本，并与这版无 MTP 对照。尚未实现 MTP，不预先承诺收益。

4K prefill 的 TTFT 相比早期实现仍有回退，作为已知问题保留；当前优先确保 decode 收益在 Speed 中成立。
