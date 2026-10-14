> 当前 MTP2 稳定分支检查点（2026-10-14）见 [交付说明](docs/strata-q2-mtp2-checkpoint.txt)及 [Speed 消费合同](integration/speed/consumer-contract-strataq2mtp2.json)。以下历史候选描述保留；最新已验收范围以该独立合同为准。

# BuildFreetokenForSpeed

正式维护仓库为公司 [siltok-ai/BuildFreetokenForSpeed](https://github.com/siltok-ai/BuildFreetokenForSpeed)。后续源码、制品合同、权重说明及证据只更新公司仓库；此前私人仓库保留历史，不再作为更新目标。

Flash Next的MTP候选保持在独立分支：见[完整接入设计](research/mtp/DESIGN.md)及[Strata实现对照](research/mtp/STRATA-MAPPING.md)。无MTP dev7制品保持冻结；正式MTP调度四组512-token匹配对应native基线，83项组件测试与12个子测试、41次基础生命周期生成通过。独立候选profile/version为 `strataq2mtp1` / `0.1.3+speed.strataq2mtp1`，新wheel已生成；尚未完成Speed服务、稳定性能和发布验收，不能继承dev7验收，也不能将开发源码覆盖作为正式制品。

当前后续候选为 `strataq2mtp2` / `0.1.3+speed.strataq2mtp2`：修复review中复现的最后in-flight窗口取消，84项组件与12个子测试通过。mtp1材料保留；新版本仍需安装后的真实取消、Speed及性能/回归验收，不属于已完成的稳定发布。

为 Silitok Speed 维护可追溯的 **FreeToken 官方源码 + 本地补丁 + Windows 构建素材**。本仓库既提供已验证的 wheel / DLL，也提供从冻结源码重新生成这些素材的入口。

当前构建候选为 **`0.1.3+speed.q2opt2`**，官方来源 `078132480ea4a0424f677a27eb22085624d331e6`，CPython 3.12 / Windows x64 / Torch 2.11.0+cu130 / CUDA 13.0 / SM120。新构建默认使用 `build/profiles/q2opt2.json`；历史 win1 含 INVALID 专家别名实验，仅保留复现。

本轮修正专家文件身份重复写死、finalize 实际字节验证、Speed 分片回执覆盖、patch 行尾与派生材料验收继承。引擎改动整理为[六组有序补丁](patches/q2opt2-series.json)，详见[审查结论与重放入口](docs/q2-material-review.md)及[q2opt2 消费合同](integration/speed/consumer-contract-q2opt2.json)。既有 wheel / patch / 验收记录不覆盖。

候选已通过来源 / 二进制验证、16 项材料工具测试、17 项引擎读取 / GPU / RowStore 测试、163 项 Speed 相关测试、完整专家流式读取及三种模型随包回归。释放后台程序后，新版本完整 Flash Next 的默认预算 HTTP、backend 长输入 / thinking / 取消复用均通过；208 个编译模块来源一致。确切范围见[追加运行验收](artifacts/q2opt2-cp312-cu130-sm120/validation/runtime-review-20261011/validation-summary.json)。初次 commit 拒绝记录保留，安装器和干净机器未验收。

Flash Next Q2 的新性能修订为 **`0.1.3+speed.q2opt1`**：按专家分组 prefill、小残余按需加载，配合 Speed 的 Q2 chunk4096，4K 原生 TTFT 约8.75→1.09秒，热6-token残余约751→91ms。完整专家和原生二进制保持不变；新wheel、来源与证据位于 `artifacts/q2opt1-cp312-cu130-sm120`，详见[优化、实测及复现](docs/q2-performance-optimizations.md)、[消费合同](integration/speed/consumer-contract-q2opt1.json)。旧版本及以下历史验收记录保留。

`feat/flash-next-q2-experts` 发布了独立的 [Flash Next Q2_0 专家候选](docs/q2-expert-candidate.md)，使用 `build/profiles/q2exp1.json` 和 [q2exp1 消费合同](integration/speed/consumer-contract-q2exp1.json)。它保留全部 48×512 专家，只替换专家来源，DRAM bank 为 31.640625 GiB；源码已通过逻辑题、两旧模型实际回归、4K 输入和 thinking/取消复用。wheel、26 DLL、冻结源码补丁与证据在 `artifacts/q2exp1-cp312-cu130-sm120`。历史 win1 制品和以下验收记录继续代表原基线。

权重获取、组合方式与自行分发见独立目录 [Flash Next 混合权重](model-weights/qwen3.8-flash-next-q2/README.md)。该目录保存来源锁、精确文件清单和分发说明，不存放大权重。

## 以下为保留的 win1 基线说明

**当前已完成 Speed 源码与随包运行目录验收。** 原两个 NVFP4 模型已经回归通过；完整 Windows `build.ps1 -SkipInstaller` 打包后，三个模型也通过 embedded Python / 干净 PATH 下的实际运行。Qwen3.8-Flash-Next 仍仅通过明确标记的 `INVALID-alias20` 专家复用实验，输出不代表完整模型质量。安装器与干净机器尚未验收。

## 从哪里开始读

| 想了解的内容 | 阅读入口 |
| --- | --- |
| 官方代码在哪里，模型原本怎样实现 | [官方源码说明](upstream/README.md)、[完整原始源码](upstream/FreeToken/) |
| 我们究竟修改了什么，为什么修改 | [改动说明](docs/changes.md)、[4 组可读补丁](patches/series/) |
| 将补丁应用后，直接浏览、开发定制源码 | [准备源码命令](#只准备源码不编译) |
| 复制素材给另一台机器，或重新编译 | [构建与复制说明](build/README.md) |
| Speed 实际使用哪些文件，怎样接入 | [Speed 接入约定](docs/speed-integration.md) |
| 已经跑了哪些验证，有什么限制 | [验收记录](docs/validation.md)、[最终随包验收](artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/)、[历史原始报告](artifacts/win1-cp312-cu130-sm120/validation/) |
| 如何持续获取上游更新、维护本地差异 | [长期维护方案](docs/maintenance.md) |

## 内容分层

```text
BuildFreetokenForSpeed/
├── README.md                       人类阅读入口
├── upstream/                       官方输入：不修改
│   ├── lock.json                   官方仓库、完整 commit、归档 SHA
│   ├── freetoken-0781324.zip        冻结源码归档，构建实际使用
│   └── FreeToken/                  同一归档展开，方便网页浏览原代码
├── patches/                        我们对 FreeToken 的差异
│   ├── series.json                 顺序、稳定编号、用途、文件清单、SHA
│   ├── series/                     Windows 运行 / 构建 / 实验 / 版本补丁
│   └── legacy/                     历史完整补丁，用来证明旧制品来源
├── build/                          生成 wheel + AOT DLL
│   ├── build-materials.ps1          Windows 构建入口
│   ├── build_materials.py           校验、准备源码、编译、生成记录
│   ├── profiles/win1.json          本次构建合同与输入 SHA
│   ├── requirements.lock.txt       71 项构建依赖版本
│   └── specs/speed-sm120.txt        Speed 需要的 26 个 AOT 规格
├── tools/                          准备源码 / 导出上游 / 校验仓库与素材
├── integration/speed/              消费约定与模型 manifest 示例
├── artifacts/win1-cp312-cu130-sm120/ 已验证制品与历史证据
│   ├── wheels/                     1 个定制 wheel，内含 3 个原生 .pyd
│   ├── kernel-cache/               26 个 AOT DLL 与匹配元数据
│   ├── build-record.json           历史构建身份、逐制品 SHA、验收范围
│   ├── validation-summary.json     历史验证摘要与 Speed 源码身份
│   └── validation/                 实际运行报告、关键日志
├── docs/                           原因、接入、验收、长期维护
└── tests/                          本仓库工具的失败边界测试
```

机器以 JSON 合同、补丁顺序和 SHA 为准；人可以从说明文档逐级跳到原代码、差异和实测证据。`upstream/FreeToken/` 保留官方原貌；我们的定制代码通过补丁生成到独立输出目录，避免在一个目录中混淆官方与本地代码。

## 官方提供什么，我们提供什么

官方输入来自 [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken/tree/078132480ea4a0424f677a27eb22085624d331e6)，固定的是当时 main 的完整 commit，源码版本为 0.1.3，**并非只按 0.1.3 tag 或 PyPI 版本取代码**。

Qwen3.8-Flash-Next 的 `qwen4_exp` 架构、QSA、NVFP4 专家、MoE offload，以及磁盘 PLE / n-gram 表方案来自官方。我们的工作是把这套能力接到 Windows 和 Speed，并为有限内存设备增加一个显式的无效模型实验。SSD 存 PLE 不是我们新发明的功能，专家复用也不是一种保持模型正确性的更低精度量化。

| 编号 | 我们的差异 | 默认行为 |
| --- | --- | --- |
| FTWIN-001 | Windows RowStore 文件读取、驱动符号加载、内存与内核加载兼容 | 支持 Windows 运行，含 SSD PLE 路径 |
| FTWIN-002 | MSVC 原生扩展、CUDA / TVM AOT 构建、兼容头文件 | 构建 Windows wheel 与匹配 DLL |
| FTEXP-001 | 只分配保留专家层，其他逻辑层复用已加载权重 | 默认不启用，双开关显式进入 INVALID 实验 |
| FTPKG-001 | wheel 版本改为 `0.1.3+speed.win1` | 与官方发行包区分，绑定本地素材 |

每组的具体代码位置、实现方式和升级处理见 [改动说明](docs/changes.md)。本仓库包含完整官方代码与完整本地 FreeToken 改动；Speed 自己的 backend / worker / 打包源码继续由 Speed 仓库维护，这里提供其消费合同和素材，不复制 Speed 的业务源码。

## Speed 实际使用的文件

| 文件 | 用途 | 当前身份 |
| --- | --- | --- |
| `wheels/freetoken-0.1.3+speed.win1-cp312-cp312-win_amd64.whl` | 安装到 Python runtime，包括 FreeToken Python 包和原生扩展 | SHA-256 `924bee8598cdd058104b8d3dd2cc4936860e092bd269616e0e832a7fbb87e5cf` |
| `kernel-cache/<spec>/<spec>.dll`，共 26 个 | 禁用 FreeToken C++/CUDA JIT 时加载的预编译算子 | 完整列表与逐文件 SHA 在 [build-record](artifacts/win1-cp312-cu130-sm120/build-record.json) |
| `kernel-cache/speed-kernel-manifest.json` | Speed 校验 Python / Torch / CUDA / SM / 规格匹配 | 同一制品集的合同 |
| `kernel-cache/build-metadata.json` | 构建来源与工具链说明 | 与 DLL 同目录保留 |

wheel 内包含 `_pinned_tensor`、`_cpu_moe`、`_row_store` 三个 `.pyd`。它们随 wheel 安装，AOT DLL 单独部署到 kernel-cache。源码 zip、补丁、依赖 lock、构建日志用于生成与追溯，不是 Speed 推理所需的模型文件。本仓库不含权重、虚拟环境或完整 Speed 安装包。

当前初始制品约为十余 MB，随仓库保留，便于首次复制直接验证；长期的大型 / 多版本制品建议放 GitHub Releases，仓库保留合同和校验信息，见 [维护方案](docs/maintenance.md)。

## 快速校验与复用

普通 CPython 3.12 和 Git 即可校验仓库，不需要 GPU、Torch、CUDA 或下载模型：

```powershell
git clone git@github.com:siltok-ai/BuildFreetokenForSpeed.git
cd BuildFreetokenForSpeed
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

校验覆盖：官方展开源码与归档一致、输入 SHA、4 组补丁与历史完整补丁等价、wheel 中的 Python 源码与补丁结果一致、26 DLL 库存与 SHA、Speed cache 合同。GitHub Actions 配置在 Windows / Linux 执行这些轻量检查；GPU 运行和完整打包验收属于另一阶段。

只需复制已有素材时，复制整个 `artifacts/win1-cp312-cu130-sm120/`，在目标机器运行：

```powershell
python -B tools/verify_materials.py D:\materials\win1-cp312-cu130-sm120
```

### 只准备源码，不编译

```powershell
python -B tools/prepare_source.py --output .\out\inspect-win1
```

得到 `out/inspect-win1/build-source/`：官方代码已经按顺序应用全部本地补丁，可以直接查看 `python/freetoken/`、`tests/`、`tools/`。同时输出补丁日志和 `preparation-record.json`。输出目录必须不存在；脚本不会覆盖旧制品或修改官方浏览目录。

### 从冻结输入重新生成

准备独立 Windows x64 CPython 3.12 构建环境、VS2022 C++ toolset 14.44、CUDA Toolkit 13.0，以及 [71 项精确版本依赖](build/requirements.lock.txt)，然后：

```powershell
.\build\build-materials.ps1 `
  -BuildPython D:\envs\ft-win1\Scripts\python.exe `
  -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' `
  -OutDir D:\builds\ft-win1-001 `
  -PreflightOnly

# 预检通过后，使用同一命令去掉 -PreflightOnly，正式编译。
```

完整的依赖安装、输出目录、失败日志、跨机器复制以及 Speed 打包参数见 [构建说明](build/README.md)。重新构建具有相同源码与版本合同，但编译工具链 / 时间等可能使二进制字节不同；以该次实际输出 SHA 为准，重新验证后才能交付。

## 已验证状态

- Qwen3.6-35B-A3B NVFP4 与 Qwen3.8-27B NVFP4：实际 Speed HTTP / SSE、取消后 worker 复用、算术与 thinking 样例通过。
- Flash Next：20 物理专家层承载 48 逻辑层，SSD PLE，Speed HTTP / SSE / 取消复用通过；进程树 RSS 峰值约 31.05 GiB。两轮短请求约 35.85 / 35.71 token/s，**只代表 INVALID 实验链路，不能作为完整模型性能或质量**。
- 最终 Speed 分支 [feat/freetoken-win1-flash-next](https://github.com/siltok-ai/siltok-speed/tree/808003f0efc70e7f43226e5d1e8c32b3704405cb)，提交 `808003f0efc70e7f43226e5d1e8c32b3704405cb`；199 项定向回归通过。此前更广 discovery 的 6 项失败在原打包基线上同样复现，历史记录保留。
- 完整运行目录重新编译 208 个 Speed 模块、重建并验证 H3、暂存定制 FreeToken 与 26 DLL，零 `.pyc`。三个模型各完成 HTTP/SSE 和 backend 短长请求、取消后复用、退出，共 6 组实际运行通过；[报告与身份记录](artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/)已封存。
- 构建发生在最终提交前，manifest 保留基线 commit + `dirty=true`，由 208 个模块源码 SHA 绑定最终代码。此次未生成安装器，也未验收干净机器或完整正确 Flash Next。

详细边界与证据入口见 [验收说明](docs/validation.md)。历史 win1 的完整 NVFP4 专家约 63.46 GiB，未在本机完整实测。独立 [Q2_0 专家候选](docs/q2-expert-candidate.md) 已于 2026-10-10 在本机通过完整 48×512 专家的 Speed 源码 HTTP/SSE 与短问题正确性验收：专家 bank 31.64 GiB、实际 worker RSS 约 33.61 GiB、加载峰值约 35.46 GiB。此候选保留 HF 主干及 SSD FP8 PLE，使用新版本材料；后续三个模型共 6 组随包验收已经通过，当前范围和证据以 [Q2_0 候选说明](docs/q2-expert-candidate.md)为准。

## 后续维护原则

完整 Strata Q2 路径的当前研究修订由 `build/profiles/strataq2dev3.json` 与 `patches/strataq2dev3-series.json` 重放，版本为 `0.1.3+speed.strataq2dev3`；dev1/dev2 源码、补丁和制品保留。dev2 修复长前缀复用的计算与归约差异，dev3 固定 GGUF PLE gate 的 FP32 归约，处理多请求 decode 经 Q8 放大的数值差异。当前已通过进入第一版 FreeToken 性能评估的正确性门槛：849 项实际长输入组件与 504 项状态检查通过，预先固定的 20 项质量对照为 FreeToken 15/20、同权重 Strata 15/20、历史混合权重 16/20，没有 FreeToken 独有失败。保留整模型长上下文数值差异及局部质量检查的范围限制，不承诺跨引擎逐位一致。当前不更新 Speed 消费合同，也不继承 q2opt2 整包验收。证据见 [研究记录](docs/strata-q2-research.txt)、[dev3 正确性进展](docs/strata-q2-dev3-correctness.txt)、[扩展任务、真实状态与长上下文对照](docs/strata-q2-dev3-expanded-correctness.txt)及[正确性门槛与质量验收](docs/strata-q2-dev3-quality-acceptance.txt)。第一版正式性能已完成，主要性能修复、MTP 与 Speed 验收仍待完成。

补充[权重使用复核与 thinking 对照](docs/strata-q2-dev3-thinking-and-model-use.txt)：有效词表、BPE 与实际聊天模板渲染通过 GGUF 对照。此前直接作答失败的三道基础题，在 FreeToken Q2、Strata Q2 与历史混合权重模型上开启 thinking 后均答对；两套 Q2 的完整输出 token 序列一致。

已完成[第一版性能与主要退化诊断](docs/strata-q2-dev3-first-performance.txt)：FreeToken 完整 GGUF Q2 约9.6 tps，同机 Strata 自动 resident 模式无 MTP 约122–126 tps中位数。trace 将首要问题定位到通用 GGUF 投影；HC 融合及小投影 GEMV 原型将32-token诊断提高到约31–34 tps，但长prefill发生退化，因此尚不形成优化版wheel或Speed消费合同。当前优先修复主要执行效率问题，并继续验证共同数值约定，性能恢复合理后再评估MTP及Speed接入。正式测量与原型门槛、历史诊断失败和资源边界均保留在追加证据中。

后续[decode 优先的结构修复实验](docs/strata-q2-decode-repair-20261013.txt)以专用压缩 MMVQ 设计验证为主：FreeToken 独立进程原型在1024槽、无MTP下，四个实际512-token请求约51.1–53.5 tps；专家搬运成为最大单项耗时。短输入冷热序列差异、prefill退化和淘汰的低效整数原型均保留，参考CUDA DLL仅用于诊断。下一步是新的 FreeToken AOT/profile/制品与必要状态/质量验收，然后优先改善专家驻留、搬运和重叠；微小矩阵差异留档，不逐项深究逐位一致。当前没有新优化wheel验收或Speed接入。

已形成[dev4 原生 decode 修复](docs/strata-q2-dev4-native-decode.txt)：独立版本重新编译 3 个原生扩展和 27 个 AOT，将压缩 MMVQ 集成到 FreeToken，脱离诊断 DLL。实际 12 个 512-token 请求无 MTP，短输入中位数 53.81、4K 51.10 tps，冷热完整输出一致；必要组件、logits、graph、前缀与取消复用检查通过。4K cold TTFT 从 dev3 4.62 秒退化至 6.90 秒，仍记录并待优化。decode 优先，当前最大单项为专家搬运约 7.3–7.5 ms/token，随后是专家计算约 3.8 ms；下一轮优先改善驻留与主要专家执行设计。此轮是独立引擎有限验收，旧模型、多请求、MTP 和 Speed 随包仍待完成。

已形成 [dev5 routed Q2 专家 decode 修复](docs/strata-q2-dev5-routed-decode.txt)：取消全缓存路由整理与 16-token decode tile，专家投影 trace 约 3.8→0.75 ms/token。实际安装新 wheel，4096 槽的 12 个 512-token 请求无 MTP，短/4K 中位数 90.71/86.78 tps；6144 槽另一组 12 次为 104.05/99.35 tps，但 commit 余量仅约 2.1 GiB，尚非资源准入配置。实际质量 15/20 与 Strata 相同、无新增失败，状态/logits/取消复用通过；短输入冷热长输出差异及 18 passed / 1 过时逐位断言失败保留，单独有界数值契约通过。下一重点为 [完整专家驻留与 host complement 设计](docs/strata-q2-expert-residency-design-20261013.txt)，持续 trace 并对照 Strata；prefill 回归、旧模型、MTP、Speed 与随包验收仍待完成。

固定上游 commit → 在独立候选分支重放小补丁 → 原生与模型回归 → 生成新的定制版本和配套 DLL → Speed 升级锁定的素材身份 → 完整包验收。上游已覆盖的改动应删除，本地差异尽量缩小；实验与正式版本逐步拆开。具体步骤与责任边界见 [长期维护方案](docs/maintenance.md)。

官方源码、版权和贡献约定保留于原目录；本地补丁及工具采用根目录 Apache-2.0 [LICENSE](LICENSE)，来源说明见 [NOTICE](NOTICE)。定制 wheel 是 Speed 发行素材，不是官方 FreeToken 发布。
