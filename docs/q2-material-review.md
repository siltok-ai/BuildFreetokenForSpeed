# Q2 材料审查与 q2opt2

本修订基于冻结官方源码 `078132480ea4a0424f677a27eb22085624d331e6`，定制版本为 `0.1.3+speed.q2opt2`。它修正来源验证和材料维护问题，保持 q2opt1 的计算、调度、专家容量与原生二进制。历史性能收益仍引用 [q2opt1 的实际 A/B](q2-performance-optimizations.md)，不能记作本修订的新收益。

## 审查结果

| 位置 | 判定与处理 |
| --- | --- |
| 引擎 `q2_source.py` 的单文件 SHA、文件尺寸和默认模型几何 | 从格式实现移除。调用方提供模型几何，准备阶段校验实际 SHA，运行时核对回执、尺寸、mtime 和完整 GGUF 目录；支持其他满足约束的完整 Q2 专家文件。 |
| 下载 / checkpoint 准备工具中的来源常量 | 统一读取 `model-weights/qwen3.8-flash-next-q2/source-lock.json`，允许显式 `--source-lock`。每份发布来源仍必须锁定，不接受“任意最新模型”。 |
| finalize 只核对旧下载回执和尺寸 | 修复。发布完成状态前重新扫描专家文件实际 SHA；确认配置路径 / 来源身份一致，以及全部 HF 分片回执覆盖和文件状态。真实同尺寸损坏文件回归测试会失败，配置不被更新。 |
| Speed 的 HF 回执覆盖检查 | 修复。回执文件集必须恰好覆盖 index 的分片；来源身份相互核对；worker 用 checkpoint 几何验证完整 bank，而不是重复 48/512/容量常量。 |
| Range 续传目录 | 按锁定 SHA 分目录，避免不同来源共享旧分块；最终整文件 SHA 校验保留。ModelScope 的 master 镜像只作传输来源，身份以锁定实际字节为准。 |
| patch 重放行尾 | 显式固定 `core.autocrlf=true`，对应已冻结的 CRLF 官方归档 / wheel，避免调用方 Git 配置改变新文件字节。用真实 Git 目录分别验证 true/false 设置；历史材料仍逐字节一致。 |
| 派生 wheel 的旧验收字段 | 新记录仅继承构建环境身份，不继承旧运行验收路径。原生复用来源由记录中的 base_materials 指定，不再写死 q2exp1。 |
| 随包验收工具的 q2exp1 版本常量 | 改为使用随包 Speed 的精确材料合同，新旧版本各自验收。 |
| 随包验收日志编码 | 隔离解释器 `-I` 忽略 PYTHONUTF8 环境变量，改为显式 `-X utf8`，避免中文日志按本机代码页输出。模型分组可独立执行，但只有三模型六组成功才算完整随包验收。 |
| wheel SHA / ABI / Torch / SM120 / MSVC / AOT 规格 | 保留。这些是经过验收的发布身份和实际编译约束；改动它们需要独立重建与硬件验收。 |
| Speed 已支持模型的几何白名单、资源准入和预算 | 保留。泛化来源文件不代表已经验收任意模型；31.640625 GiB 完整专家 bank 仍由实际模型几何得出。 |
| 小残余 prefill 的缓存容量 | 无需修复。`OffloadMoeCache.validate_rebuild` 已要求 cache_size >= num_experts，而该路径的路由数 <= num_experts/3。新增容量兜底没有可触发的收益。 |

读取 checkpoint 和运行时检查的尺寸 / mtime 是已验证不可变权重的廉价状态检查；完整 SHA 在下载与 finalize 中执行。它们不承诺抵抗刻意伪造回执、还原 mtime 的文件替换。

## 当前补丁序列

完整目录为 [q2opt2-series.json](../patches/q2opt2-series.json)，冻结输入为 [q2opt2.json](../build/profiles/q2opt2.json)。顺序有依赖；性能补丁可单独审阅，不与平台代码或包版本混合。

| 顺序 | 补丁 | 保留理由与删除条件 |
| --- | --- | --- |
| 1 | Windows runtime | Windows RowStore SSD 行读取、驱动符号与 kernel loader、host bank 批量读取、无 POSIX fadvise 时的平台分支。上游提供等价 Windows 行为并回归后移除。 |
| 2 | Windows build | MSVC 原生扩展、Windows AOT 与兼容头；只服务冻结工具链。上游构建入口覆盖该组合后移除。 |
| 3 | Q2 routed experts | 原生 64 元素 / 18 字节 GSQ Q2_0 bank、量化注册、完整专家来源与向量算子。纯 GPU offload、TP1、BF16、SiLU、输出侧路由权重；不包含 CPU / resident / hybrid 或完整 GGUF 主干加载。上游支持等价格式后迁移。 |
| 4 | Grouped prefill | 按专家分组的 BF16 矩阵 prefill，独立 GPU 数学参考覆盖 padding、空专家和集中路由。保留 q2opt1 实测有效的阈值 / tile；有上游同等实现后移除。 |
| 5 | Small residual prefill | Q2 小残余按实际路由加载专家，复用原 decode LRU；完整长 prefill 保留既有路径。独立性能理由见 q2opt1 A/B，不能扩大到未验收量化格式。 |
| 6 | Package identity | 专用本地版本，防止官方包与定制 wheel 混用；此补丁不向上游贡献。 |

当前序列不包含 `FTEXP-001` 专家别名，也不包含 INVALID 实验脚本。`win1`、`q2exp1`、`q2opt1` 的序列、SHA 和已验收材料封存不变；历史目录里的 active 标记仅描述当时构建，不能据此加入当前包。

## 重放与打包

```powershell
python -B tools/prepare_source.py --profile build/profiles/q2opt2.json --output D:\builds\q2opt2-source
python -B tools/validate_python_revision.py --profile build/profiles/q2opt2.json --materials artifacts/q2opt2-cp312-cu130-sm120
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

Speed 取得 [消费合同](../integration/speed/consumer-contract-q2opt2.json) 中成对的 wheel/cache，以 `-FreetokenWheel` 和 `-KernelCacheDir` 显式传入其打包入口。不得重命名旧 wheel、覆盖旧版本源码或混用旧 manifest。

当前原生 / AOT 输入与 q2opt1 完全一致，使用 allowlist 验证后派生 Python wheel；任何原生、AOT 或工具链输入变化会拒绝复用。重新生成的命令：

```powershell
python -B build/derive_python_materials.py --profile build/profiles/q2opt2.json `
  --base-profile build/profiles/q2opt1.json --base-materials artifacts/q2opt1-cp312-cu130-sm120 `
  --output D:\builds\q2opt2-materials
```

仅库存或离线重放通过不代表 GPU / 产品验收。真实运行证据追加在新材料的 validation 目录，记录确切 wheel、源码哈希和范围；新记录不继承旧验收。

本轮[验收摘要](../artifacts/q2opt2-cp312-cu130-sm120/validation/review-20261011/validation-summary.json)记录：16 项材料工具、17 项引擎来源 / GPU / RowStore、163 项相关 Speed 回归通过；真实专家全部读取完成；三种模型中的两种旧模型共四组随包回归通过，208 个编译模块来源无差异。全量 Speed unit 的 13 个失败在未修改 q2 基线上全部重现。

新版本 Flash Next 默认准入需要约 51 GiB 可用 commit，目前整机余量不足，初次实际 HTTP 请求按预期被拒绝。未降低预算 / 余量、调整系统 pagefile 或停止他人的程序。完整模型、安装器和干净机器状态均保持未验收；q2opt1 历史成功记录继续独立保留。
