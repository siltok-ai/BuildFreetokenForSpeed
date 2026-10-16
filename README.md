# BuildFreetokenForSpeed

本仓库为 [Silitok Speed](https://github.com/siltok-ai/siltok-speed) 提供 FreeToken 的冻结源码补丁、Windows wheel/AOT、权重准备工具和消费合同。官方浏览源码 `upstream/FreeToken/` 保持原貌；应用 backend、worker 和 HTTP 留在 Speed。

当前版本由 **[integration/speed/current-release.json](integration/speed/current-release.json)** 唯一选择。main 对应 Speed 最新 Flash Next Strata Q2 实现；更新前的探索分支 `explore/flash-next-strata-q2-mtp` 保留开发历史。

| 项目 | 当前选择 |
| --- | --- |
| FreeToken | `0.1.3+speed.strataq2perf8` |
| profile / 有序补丁 | `build/profiles/strataq2perf8.json` / `patches/strataq2perf8-series.json`（48组） |
| 成对制品 | `artifacts/strataq2perf8-cp312-cu130-sm120/`：wheel内3个native模块，27个AOT DLL |
| 运行合同 | Windows x64 / Python 3.12 / Torch 2.11.0+cu130 / CUDA13 / TVM FFI 0.1.13.post3 / SM120 |
| Speed 对应分支 | `integrate/flash-next-strata-q2-mtp-5070ti`，验收提交 `480c1ae26b170b8b2c7e1359a828dcfb88271844` |
| MTP | **默认关闭**；显式 `paired_cjk` 在完整词表3草稿与CJK词表2草稿之间按整轮收益选择；原 `full` 保留，非greedy原采样 |

本轮5070 Ti优化已完成普通wheel、Speed源码及实际编译保护载荷的质量、资源、生命周期和HTTP验收。当前合同明确区分暂存载荷与真实安装，不以旧安装记录替代本轮证明；旧质量集15/20、thinking3/3，无新增失败，保留间歇长输出已知问题。当前测量、复现命令和范围见 [perf8交付记录](docs/flash-next-5070ti-perf8.txt)。历史版本、补丁与验收资料保持不可变。

以下保留早期候选演进记录，当前选择以上方唯一指针为准。

5070 Ti 性能分支另有显式候选 `strataq2perf2`，包含缓存借用、搬运重叠与交错投影；默认当前指针保持上述已验收版本。实测范围、原生重建边界和待完成验收见 [性能候选记录](docs/flash-next-5070ti-performance.txt)。
后续独立候选 `strataq2perf3` 增加紧凑MTP事务、快照容量配置、在线热点所有者交换和宽tile，原生材料字节沿用perf2；已测范围与尚未完成的Speed验收见 [第三轮记录](docs/flash-next-5070ti-perf3.txt)。
独立候选 `strataq2perf4` 提前将原始压缩embedding放映射主存、支持专家旧热度衰减并复用跨层准入JIT；源码开关MTP质量与状态扩缩容通过，普通wheel/Speed验收边界见 [第四轮记录](docs/flash-next-5070ti-perf4.txt)。

后续 [Q2执行研究检查点](docs/flash-next-5070ti-q2-execution.txt)保存了有收益的字节置换解码、GPU所有权元数据发布及缓存副本晋升原型，以及未采用的CPU协作/原生聚合实验。OFF原生性能已进入同Q2 Strata参考范围，ON仍有差距；这些原型尚未形成新交付材料。

独立候选 `strataq2perf5` 已将有收益的Q2解码、所有权发布和有界PLE行缓存冻结为40组补丁；行存储扩展真实重编，另外两个扩展与27项AOT复用。[第五轮记录](docs/flash-next-5070ti-perf5.txt)区分了源码、普通wheel和Speed验收范围，当前默认指针仍保持已验收版本。

后续研究检查点保存了 [MTP词表原型与Strata反向消融](docs/flash-next-5070ti-mtp-review.txt)：对照CJK/英文/完整词表、同步提交，以及草稿KV ring、窗口流水和长度策略的实际接线。该检查点尚未形成新的正式profile或Speed交付版本。

独立候选 `strataq2perf6` 借用整个prefill临时工作区，扩大专家缓存后开关MTP五项吞吐均改善。pinned扩展实际重编，原始Q2/FP32语义与完整专家保留；普通wheel、故障隔离及状态扩缩容已验收，[第六轮记录](docs/flash-next-5070ti-perf6.txt)说明资源配置与尚未完成的Speed验收。

## 使用当前版本

```powershell
# 验当前指针、材料和 MTP 身份，不启动模型。
python -B tools/verify_current_release.py

# 默认重放当前版本；要复现旧版本可显式指定 --profile。
python -B tools/prepare_source.py --output D:\builds\freetoken-mtp2-source

# 权重准备：先计划，再按固定 revision 下载并验真，生成独立 checkpoint。
python -B tools/flash_next_weights.py plan --root D:\models\flash-next
python -B tools/flash_next_weights.py download --root D:\models\flash-next
python -B tools/flash_next_weights.py prepare --root D:\models\flash-next

# 仅在需要 MTP 时提取31份 BF16来源并转换；不下载360GB完整BF16主模型。
python -B tools/flash_next_mtp.py fetch --root D:\models\flash-next
python -B tools/flash_next_mtp.py build --root D:\models\flash-next
python -B tools/flash_next_mtp.py verify --root D:\models\flash-next
```

完整 GGUF 两片、tokenizer与元数据、MTP源文件和派生数据按发布者/revision/用途分目录。GGUF是文本全权重路径，HF仅提供固定配置与tokenizer；不加载RadixArk HF主干。权重大文件不进Git。搬迁后执行 `flash_next_weights.py refresh` 重新完整验真并生成目标机器的绝对回执；可搬迁来源描述保留相对路径。FT2引擎本身要求绝对运行路径，不用修改旧引擎绕过校验。

[当前交付流程](docs/flash-next-current-delivery.md)包含安装wheel、明确选择新cache、配置开关、同预算性能对照、目录搬迁和失败处理。性能preset需要显式选择：原SM120/24GiB方案保持原样，另有5070 Ti/16GiB的已测容量与余量档，不替换全局安全策略。[5070 Ti交接实测](docs/flash-next-5070ti-acceptance.md)记录源码HTTP短/4K约71.76/68.94tps及MTP对照，包含真实安装/卸载验收及未解决长输出风险。

## 构建和验证

重新编译及环境准备见 [build/README.md](build/README.md)。输入通过当前指针选择；旧版用 `--profile build/profiles/q2opt2.json` 等显式重放。编译工具链变化可能产生不同二进制SHA，需要另存制品、重新验收，不能覆盖已接受wheel或改版本标签。

```powershell
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
python -B tools/verify_mtp2_consumer.py
python -B tools/verify_flash_next_5070ti.py
```

上述离线检查不等价于GPU/整包验收。历史5090 D v2结果：无MTP最佳历史HTTP短/4K中位109.88/104.02tps；MTP六次各512个实际交付token，中位106.54/109.58tps，同容量关闭101.79/97.43。20题非thinking15/20且输出IDs匹配已接受Q2基线，thinking三题3/3；旧27B/35B、取消复用和故障重启已通过安装版本验收。详细证据见 [消费合同](integration/speed/consumer-contract-strataq2mtp2.json)、[原始检查点](docs/strata-q2-mtp2-checkpoint.txt)。不是所有设备的100tps保证。

新增[可搬迁交付验收](research/mtp/evidence/portable-delivery-20261014-v1/README.md)：固定元数据网络下载、完整两片校验、31份MTP重新转换、异路径刷新、Git导出29项测试与已安装Speed开/关各两次512-token推理均通过。`python -B tools/verify_flash_next_delivery.py`交叉核对封存原始记录。

5070 Ti新Cython整包已真实安装验收并清理，按用户决定带已知长输出风险接受，详见当前消费合同；独立整合分支亦已完整构建、实际安装回归并清理；干净操作系统尚未验收。第三方有版本锁但未提供完整离线wheelhouse。源码/格式/许可与维护规则见 [upstream](upstream/README.md)、[NOTICE](NOTICE)、[维护说明](docs/maintenance.md)。

旧版本保持可回滚：`q2opt2` 为 HF主干+Q2专家混合模型；`strataq2dev7` 为无MTP全GGUF检查点；`strataq2mtp1` 为首次MTP候选。旧入口 `integration/speed/consumer-contract.json` 代表历史win1，不是当前选择。历史README保存在 [历史记录](docs/history/readme-before-current-20261014.txt)。

perf6后续源码检查点已通过真实TCP、冷采样与取消/卸载重载验收。公开preset使用更稳妥的OFF5760/ON4992永久专家；上文普通wheel的较大greedy容量不等于交付容量。精确测试范围、冷JIT协议修复和性能见[第六轮记录](docs/flash-next-5070ti-perf6.txt)，当前默认材料指针在最终编译验收前继续保持历史版本。
