# 长期维护 Speed 对 FreeToken 的使用

目标是持续获得官方模型与性能更新，同时保留 Speed 的小范围平台适配和优化；每次升级都有明确来源、可读差异、可生成制品、旧模型回归和回滚能力。本仓库作为 Speed 之外的定制供应层，避免把修改散落在某台机器的 site-packages 或无法重建的 wheel 中。

当前默认 profile 为 `q2opt2`，有序目录为 `patches/q2opt2-series.json`，已经排除专家别名实验。六组补丁的必要性、来源身份修复、可移除条件和实际验证见 [Q2 材料审查](q2-material-review.md)。下文 win1 迁移和 normal/research 规划保留历史背景；当前 validator 同时保留 win1 等价证明，并分别核对 q2opt1 / q2opt2 的实际 profile 与原生来源。

## 三个边界

| 层 | 负责什么 | 稳定交付 |
| --- | --- | --- |
| 官方 FreeToken | 架构、权重读取、量化、调度、算子等上游能力 | 完整 commit 与许可证 |
| 本仓库 | 少量平台 / 引擎补丁、冻结构建输入、定制版本、材料生成与测试证据 | wheel + matching cache + provenance |
| Speed | 应用 backend、worker、HTTP/SSE、资源准入、模型 manifest、产品打包 | 固定材料身份的应用版本 |

能在 Speed adapter 实现的接口兼容优先留在 Speed；能通过 FreeToken 现有配置或扩展点实现的行为优先用扩展；确实需要改变核心分配 / 原生 IO / 内核时才维护 engine patch。避免把服务协议、UI 和业务逻辑搬进上游引擎。

## 上游更新：候选先于替换

1. 在单独的上游 checkout 获取官方更新，选择完整 commit；只读取官方，不自动向官方仓库推送。
2. 为新候选建立本仓库分支，保留当前 Speed 使用的已验收 profile / 制品。
3. 用导出工具生成新归档与浏览树，检查上游版本、架构、API、依赖和 native 模块变化。
4. 对每个稳定编号补丁判断“仍需要 / 官方已实现 / 改用扩展 / 放弃”，重放并人工处理冲突。
5. 冻结候选补丁、依赖 lock、规格、工具链和输入 SHA，赋予新的定制版本号。
6. 原生与模型回归通过后生成不可变制品，再由 Speed 独立升级消费合同。

示例导出命令：

```powershell
# FreeToken official checkout 中先获取所需 commit，再执行。
python -B tools/export_upstream.py `
  --repo D:\upstream\FreeToken `
  --commit <完整40位commit> `
  --output .\out\upstream-candidate
```

导出使用 `git archive <commit>`，不包含上游 checkout 的未提交修改；输出候选 `candidate-lock.json`、zip 与原始浏览树，不修改 win1。审核后纳入新的 upstream/profile 位置。producer 支持 profile 输入，但目前编译逻辑固定 win1 工具链，改变 ABI / GPU 等还需相应更新 producer。

当前验证器刻意绑定 win1，并用历史补丁证明第一次迁移正确。未来新增版本时应将候选校验推广为 profile 参数和对应 provenance 规则，保留 win1 的等价检查，不为了让新补丁通过而重写历史。

## 补丁如何持久迭代

`patches/series.json` 是有序目录，包含稳定编号、用途、类别、状态、上游处理状态、完整修改文件表及 SHA。当前按文件拆分的四组在本次来源验证中完全等价；后续新增需求应尽量一项理由一组补丁。

每个变更应同时更新：

- 修改原因与触发条件，涉及官方哪些路径。
- 是否正常启用、配置边界、数值或资源语义的影响。
- 失败前 / 修复后的验证，以及实际硬件或性能证据。
- 与上游实现的关系、可删除条件、冲突处理理由。
- 补丁 SHA、profile 输入 SHA、定制修订与发布记录。

上游已经提供同等能力时，删除对应本地差异，并验证两者行为等价；不要因为 patch 还“能 apply”就保留冗余实现。API 冲突要在独立候选源码中处理，不在安装目录临时修。补丁状态调整和冲突决策进 Git commit / PR，文档保留关键理由，避免庞大的逐命令流水账。

Windows IO、MSVC build 等通用适配可考虑向上游贡献，从而减小长期维护负担；提交前遵守官方贡献政策并由人类理解和验证，是否提交由用户单独决定。专用无效模型实验通常适合留在本地研究 profile。

## 正式与研究版本分开

当前 `0.1.3+speed.win1` 为历史已验收的混合源码：Windows 支持和默认关闭的专家复用实验同时存在。保留它用于复现，不应事后删掉代码却继续使用同一版本 / SHA。

后续建议维护：

| profile | 差异 | 可用于哪些验收 |
| --- | --- | --- |
| normal | 上游 + 必要 Windows / 构建补丁 | 正确模型、应用发布候选 |
| research-alias | normal + FTEXP 实验补丁 | 有限资源执行链路研究，INVALID 输出 |

版本号可用 `0.1.3+speed.win2` 与 `0.1.3+speed.research1` 等本地版本标识。它们是示意命名，并非已构建制品。遵循 [Python 本地版本约定](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers)，同时让 Speed 锁定**精确版本和 SHA**，不要依赖版本大小推断正常 / 研究版本兼容。

正常 worker 无论采用哪种材料都应清理 INVALID 环境；研究条目显式开关、专用模型 ID、专门 checkpoint 校验和预算。去掉实验代码后的 normal wheel 必须重新编译并回归，即使 ABI 表面相同。

## 构建矩阵与可复现程度

一次材料身份至少固定：

```text
官方完整 commit
+ 有序补丁与 SHA
+ 依赖版本 / 必要时依赖 wheel SHA
+ Python ABI / OS / Torch / TVM FFI
+ CUDA / MSVC / GPU SM / AOT spec
+ 构建工具自身 SHA
= 一个定制材料版本 + 实际 wheel / DLL SHA
```

同一源码不等于同一二进制。更换 Torch、TVM、CUDA、SM 或 native 源码后，重建 wheel / cache；不能只复用旧 DLL 并修改 manifest 的版本字符串。不同平台 / 架构应有独立 profile 与制品目录。

当前冻结源码和补丁、锁定发行包版本、记录工具链，支持来源与构建过程复现；不保证跨机器 bit-for-bit 一致。要达到长期完全离线生成，需要保存编译器安装版本、SDK、第三方 wheelhouse 和每个文件的 hash。71 项 lock 只是 FreeToken 构建闭包，Speed 的完整依赖仍另行审计。

## 验证分层

| 阶段 | 验证内容 | 当前自动化 / 状态 |
| --- | --- | --- |
| 来源检查 | snapshot/zip 一致、输入 SHA、patch 可应用、wheel 源码与制品身份 | 仓库 validator，配置 Windows / Linux CI |
| 工具失败边界 | 坏输入拒绝、路径逃逸拒绝、旧输出保留、patch 失败立即停止 | 本仓库 unittest |
| 编译 | native `.pyd`、26 DLL、工具链 / ABI / cache 合同 | Windows producer + preflight |
| 原生行为 | row-store 数值、SSD IO、同步 / wait、专家分配与资源 | 需要目标设备实测 |
| Speed 源码运行 | 老模型 + 新模型相应范围、同步 / SSE / reasoning / cancel / worker 退出 | 本次已做，证据保留 |
| 随包运行目录 | Speed 编译、embedded Python / 干净 PATH、随包 native/cache、三模型实际运行 | 已完成，见 [证据](../artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/) |
| 安装发布 | 安装器安装 / 运行 / 卸载、干净机器 | 尚未做 |

修性能需同硬件、同完整权重、同输入长度与 batch 的 A/B；INVALID-alias 不能与正确模型直接比较。新 shape 必须经过真实加载和生成，库存 / import 检查不能发现所有缺项；本次 dense 两个遗漏是具体例子。

CI 可以提前发现 hash 漂移、源码误改、patch 失效和制品混用。普通 hosted runner 不能证明 Windows GPU 推理通过；未来可增加独立 GPU runner 或人工验收报告，避免把硬件测试放到每次文档 PR 的阻塞路径。上游更新监视最多生成候选提醒 / PR，不应自动合并并替换 Speed 的材料。

## 发布材料与 Speed 升级

初次仓库保留十余 MB 已验证制品，方便直接复用；未来多版本、大制品用 [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) 存储，仓库只保留材料 manifest、SHA、build-record、输入及重要证据。

每份 release 建议包含 wheel、完整 kernel-cache 压缩包、逐文件 SHA、构建 profile / commit、补丁身份、工具链和验收报告。发布版本绑定本仓库明确 commit；上传后不替换同名素材，新修订采用新版本 / release。

Speed 仓库保存一个材料 lock：定制版本、release 或材料目录、wheel SHA、cache manifest SHA、上游 commit 和定制仓库 commit。下载 / 复制后先核对 hash，再打包；构建产出的 Speed 发布说明记录其材料身份。

当前 Speed 的精确 wheel SHA 锁已生效；后续完整集中材料 lock 仍按上述方案演进。材料仓库新增验收文档不会使已验收的 wheel/DLL 身份变化，Speed 本轮继续引用材料 commit `12bdf743bbf4725fc25468204313cb2f43733758`。新增验收记录反向绑定最终 Speed commit，避免为了同步文档制造两仓库 commit 循环。跨机器同源码重编译字节变化，需要新的材料记录、验收和 Speed SHA 更新；引擎源码变化则同时增加定制版本修订。

由 Speed 在独立分支更新 adapter 与 lock，执行两个历史模型回归，再接入新模型并完成完整包验收。不要一边改变 FreeToken/Torch/Speed 多处，一边继续覆盖同一个“最新 DLL”目录，使问题无法归因。

## 回滚与关键问题记录

旧材料目录和 Speed 旧发布 lock 都保留。出现回归时，Speed 回到旧 adapter / lock / 已验收包，不向旧 wheel 内覆盖个别新源码。用新环境并存验证，避免污染生产 venv。

关键问题记录“触发条件 → 根因 → 最终处理 → 证据 → 剩余边界”。本次主要是 Windows SSD 原生接缝、专家真实分配、AOT dense 规格遗漏、API / worker 协议、完整依赖闭包与无效模型验收边界。日常维护无需记录每条命令；源码 commit、材料 SHA 和原始报告负责精确追溯。

本仓库负责可复制的来源与资料；用户桌面的探索记录继续承载本机任务背景和决策。新的正式发布不能只依赖桌面文档，必须将其关键验收结论与材料身份同步到版本库。
