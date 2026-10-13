# FreeToken Strata Q2 dev7

dev7 修复 Speed 首次接入发现的真实兼容性缺口：原 FreeToken 的短 prefill 优化会将最多 16 token、每 token 10 个专家的请求交给 decode 专家路径，dev6 的驻留入口却仅接受 128 条路由。新入口接受最多 256 条路由，并用实际 160 条路由的 GPU 测试覆盖去重、搬运和图重放；没有关闭原 sparse-prefill 优化。dev6、历史 wheel 和两个仓库 main 均保留。

新 profile 为 `build/profiles/strataq2dev7.json`，补丁 `FTSTRATA-007`，材料 `artifacts/strataq2dev7-cp312-cu130-sm120`。wheel SHA256 为 `32c54404cbd61a435d7fb76e7c3f2185d2c08a92959ebf1462a244f9e635ba42`。相对 dev6 仅修改驻留入口、版本和实际 GPU 测试三个文件。原生三个模块和 27 个 AOT 内核继续逐字节复用 dev4；排名和 Q2 算术未变。

独立安装验证通过：456 个安装文件身份匹配；24 个组件测试；真实模型 16-token prefill；前缀分支恢复、稀疏长上下文、取消复用、拒绝危险专家池 resize 和 KV-only 重建；662 项状态组件对照误差为零，12 项稀疏选择一致。20 道质量题为 15/20，完整输出与历史版本一致。原始 state-v2 失败记录保留：测试对截断的聊天头错误要求生成满两个 token，实际自然 EOS；state-v3 遵守 EOS，仅检查真正执行了 16-token prefill，未修改引擎。

无 MTP，固定配置沿用 dev6：9216 永久专家、1024 动态槽、1024 prefill 槽，总数 11264；KV8192、图 batch1、最大 prefill4096、memory_ratio0.9、单请求、disk PLE。短输入及 4K 输入各六次，每次实际输出 512 token，decode 中位数为 112.63 / 103.42 tps，最低为 110.19 / 102.57 tps。十二次完整 token 输出与冻结原型及历史 dev5 相同。最小系统 commit 余量约 6.48 GiB。

NVML 整机峰值 25,242,763,264 字节，最小空闲 400,162,816 字节；这是驱动整机口径，不等同于 Torch reserved。当前驻留配置显存余量小，默认保守的 Speed 资源策略可能拒绝加载；不得隐藏峰值或全局削减安全余量。后续实际 Speed 请求需要独立记录峰值，并显式选择实验配置。

证据索引位于材料的 `validation/protected-residency-20261013/index.json`，57 个复制文件、2 个大型原始引用，索引 SHA256 为 `807077c7814a5ee1071751ce70256ca5caac3478c6cf376df15b27adf54e9227`。源码重放、材料与原生来源检查由 `tools/validate_sparse_residency_revision.py` 执行，旧 dev6 的严格检查未改动。

此提交只冻结独立引擎验证，Speed backend/HTTP、旧模型、打包及 MTP 验收仍待完成。先保存可运行的无 MTP 基线，再在独立修订评估 MTP：Q2 GGUF 本身没有预测头，需要另取固定 revision 的 MTP 张量，验证草稿接受率、拒绝回退、GDN/conv/KV/PLE 状态、前缀复用、取消和资源预算，比较真正交付 token/s。prefill 的历史 TTFT 回退仍是已记录问题，decode 优先。
