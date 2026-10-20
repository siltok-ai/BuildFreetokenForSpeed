# QSA Python/Triton 候选

`strataq2qsaexp1` 是独立实验 profile。`current-release.json` 仍选择已验收的 portable1；本候选不代表当前正式交付已经切换。原有补丁、wheel、原生扩展与 AOT 不覆盖。

候选在 GGUF 的固定选择顺序与 FP32 归约分段上做三项修改：

- `qsa_sparse_paged_attention(..., causal_positions=...)` 根据 GPU 上的位置跳过整块填充迭代。调用者必须保证每行索引数组在第 `position + 1` 项及之后全是无效项；该界限以内允许存在无效页导致的空项。原有 split 分界及有效 tile 的运算顺序保留；不传参数的调用继续原路径。
- 可见完成块不超过 top-k 容量时，top-k 直接按列号压缩有效项，不做 radix 排名；仍排除无效分数并补 `-1`。
- QSA 在全部可见完成块均能入选时，跳过不影响选择集合的点积评分，保留页有效性检查。该优化以模型激活有限为前提，`FREETOKEN_QSA_DENSE_SCORE_SKIP=0` 可以关闭。长上下文超过预算后仍逐查询评分、选择；压缩键与 pending ring 始终更新，以保证后续跨过预算边界时可读。

不共用不同查询的稀疏选择，不调整 index budget，也不修改 GDN 紧凑状态实现。MTP 的完整状态控制组使用已有 `engine.extra.mtp_compact_state: false`，独立于上述 QSA 修改；需要在目标机器核对增加的状态显存。

## 构建与离线检查

在仓库根目录使用配套 Python 环境执行：

```powershell
python -B build/derive_python_materials.py --profile build/profiles/strataq2qsaexp1.json --base-profile build/profiles/strataq2portable1.json --base-materials artifacts/strataq2portable1-cp312-cu130-sm120 --output out/strataq2qsaexp1
python -B tools/validate_python_revision.py --profile build/profiles/strataq2qsaexp1.json --base-profile build/profiles/strataq2portable1.json --materials out/strataq2qsaexp1
python -B tools/verify_portable_consumer.py --contract integration/speed/consumer-contract-strataq2qsaexp1.json --speed-root <匹配Speed源码目录>
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

这是允许复用原生模块和 27 项 AOT 的 Python/Triton 修订；没有修改 C++、CUDA 头文件、ABI 或 AOT 规格。`out/` 为已有忽略目录。实机实验可从另一份同 profile 的完整新构建复用二进制，但应单独验证三份原生模块及全部 DLL 与该实测基线逐字节一致，不把另一份构建的验收结果直接移用。

离线检查不执行模型。实机验证需要覆盖短/长上下文、2048 预算边界、无效页、改变 CUDA 图输入、MTP OFF/ON、原始 token 一致性，以及相同专家/KV预算和实际输出长度下的计时；实验脚本与测量记录放在仓库外。
