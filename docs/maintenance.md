# FreeToken定制与Speed消费维护

当前选择读取 [current-release.json](../integration/speed/current-release.json)。profile固定上游归档、补丁顺序与输入SHA；Speed的release固定实际wheel、缓存和供应提交。实际执行源码必须由profile重放，不在 `upstream/FreeToken` 或共享site-packages临时改代码。

## 职责和修改位置

| 层 | 负责内容 |
| --- | --- |
| 官方snapshot | 只读来源与许可证；完整commit和zip身份 |
| 本仓库 | FreeToken引擎定制、native/AOT构建、材料来源、权重准备与证据 |
| Speed | 模型manifest、配置准入、worker协议、HTTP、资源调度和产品打包 |

共享格式解析放在checkpoint层，避免kernel反向依赖具体模型。配置到引擎字段的映射在Speed集中维护。普通导入放文件顶部；CUDA初始化顺序、原生扩展启动和循环依赖所需的延迟导入保留并说明理由。不要为整齐而破坏父进程无CUDA上下文的设计。

只在真实信任边界校验：外部配置、权重身份、ABI、原生内存所有权和事务隔离仍需检查；同版本私有IPC不重复验证已构造字段。禁止用默认值或继续运行掩盖错误。测试优先覆盖状态迁移、真实字节、资源释放和数值语义，避免自增计数、getter和源文件包含字符串的测试。

## 新修订

1. 用 `tools/prepare_source.py --profile <base> --output <新目录>` 重放候选。
2. 按修改原因追加有序补丁；已接受patch、profile、wheel和证据不可覆盖。
3. 更新新catalog与profile的SHA和独立版本。运行实际失败前/修复后回归用例。
4. 按影响范围构建或推导新材料；执行源码、普通wheel和硬件检查，保留原始报告。
5. 先提交供应材料，Speed再锁定此commit；消费证据反向引用最终Speed提交，避免循环更新commit。
6. 最后更新当前指针及其输入SHA，写明各层验收范围。新源码不能继承旧安装资格。

本轮review1是相邻perf8的Python修订；`build/derive_python_materials.py` 允许的源码边界明确，3个native模块和27项AOT逐字节复用。`validate_python_revision.py --base-profile <perf8> ...` 重放两份源码、重算差异并验证复用边界。涉及row-store/pinned C++时只能走实际选择性重编；公共编译接线保留各自来源/回执。涉及其他native、AOT规格或ABI必须完整重建，不能扩大允许清单以跳过重编。

## 验证和性能

必须运行 `python -B tools/validate_repository.py` 和 `python -B -m unittest discover -s tests -v`。validator保留win1迁移等价证明和后续各历史版本校验。它检查已封存的硬件报告，不是在本次重新运行这些历史模型。

硬件验证依次覆盖：实际安装wheel的GPU组件、完整权重质量/状态扩缩容、Speed worker/HTTP取消重载、普通Speed wheel、保护编译载荷，以及需要时的真实安装/卸载。干净OS是独立范围。旧27B/35B模型按改动影响回归；设备装不下的模型不宣称通过。安装/服务变化和重要交付检查点再做安装，不反复污染本机。

性能对照固定设备、依赖、完整权重、请求、512个实际交付token、专家预算与策略。串行运行模型，不与编译竞争CPU；保留TTFT、冷热前缀、decode、输出IDs和资源监督。策略使用实测时间可能选择不同草稿窗口，长输出也可能已有不稳定；报告实际变化，不把均值或一条跑分写成所有场景保证。无效专家alias输出不得用于性能结论。

来源/构建闭包锁不等于完整离线wheelhouse。依赖、CUDA Toolkit、编译器、GPU架构改变时，使用新profile与材料，记录新工具链并重新验收；不能把Torch的cu130标签当成实际nvcc版本。

## 升级和回滚

上游候选通过 `tools/export_upstream.py --repo <checkout> --commit <完整commit> --output <新目录>` 导出，只读官方仓库。逐项判断本地patch是否仍需要，处理冲突，固定新snapshot和输入后按上述流程验收。不得自动向官方FreeToken提交；遵守其贡献政策。

回滚整体恢复旧Speed adapter/release和匹配材料，不往旧wheel覆盖新文件。保存原因、影响、证据与边界即可，精确操作由commit、SHA和报告追溯。[本轮review](flash-next-review-20261017.txt)、[当前交付](flash-next-current-delivery.md)与不可变 [旧维护说明](history/maintenance-before-review-20261017.txt) 分别提供当前结论、运行步骤和历史背景。
