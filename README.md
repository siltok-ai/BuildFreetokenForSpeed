# BuildFreetokenForSpeed

为 [Silitok Speed](https://github.com/siltok-ai/siltok-speed) 提供 FreeToken 的冻结补丁、Windows wheel/AOT、权重准备工具与消费证据。官方浏览源码 `upstream/FreeToken/` 保持原貌；应用 backend、worker 与 HTTP 在 Speed 仓库。

当前版本由 [current-release.json](integration/speed/current-release.json) 唯一选择，包含 profile、有序补丁、材料目录、preset 与验收合同。不要按文件名排序选择版本，也不要把 `consumer-contract.json` 当作当前合同。Speed 分支为 `integrate/flash-next-strata-q2-mtp-5070ti`，应用的 `qwen36/release.json` 另锁定 wheel SHA、配套缓存和供应提交。

本轮 [综合 review](docs/flash-next-review-20261017.txt) 在 perf8 的48组补丁之后追加3组，生成独立 `0.1.3+speed.strataq2review1`。修复MTP读取校验竞态，共享GGUF解析和停止条件，整理事务/策略/热点缓存代码，保留原有数值内核。新版本普通wheel、Speed源码、保护编译载荷已独立验收，完整安装器已构建；本修订未实装、未验干净OS。精确范围由自己的合同声明；perf8的真实安装证据仍在 [历史交付](docs/flash-next-5070ti-perf8.txt)，不能替代新代码的验收。

| 运行输入 | 合同 |
| --- | --- |
| 平台 | Windows x64 / CPython3.12 / SM120；本轮RTX5070 Ti 16GB |
| 依赖 | Torch2.11.0+cu130 / triton-windows3.6.0.post26 / apache-tvm-ffi0.1.13.post3 |
| 材料 | wheel内3个native模块 + 配套27项AOT DLL；逐文件SHA不可变 |
| 主模型 | 固定GSQ-RCO Q2完整两片GGUF；原始Q2专家、分段量化主干、完整主词表 |
| MTP | 默认关闭；显式 `paired_cjk` 或 `full`，独立31份BF16来源张量 |

## 使用

[当前交付流程](docs/flash-next-current-delivery.md) 包含安装配套wheel、完整权重/MTP准备、目录搬迁与5070 Ti配置。

```powershell
python -B tools/verify_current_release.py
python -B tools/prepare_source.py --output D:\builds\freetoken-current
python -B tools/flash_next_weights.py plan --root D:\models\flash-next
python -B tools/flash_next_weights.py download --root D:\models\flash-next
python -B tools/flash_next_weights.py prepare --root D:\models\flash-next
# 仅MTP开启时需要
python -B tools/flash_next_mtp.py fetch --root D:\models\flash-next
python -B tools/flash_next_mtp.py build --root D:\models\flash-next
python -B tools/flash_next_mtp.py verify --root D:\models\flash-next
```

主模型GGUF路径与HF元数据分开：HF只提供固定配置/tokenizer，不加载RadixArk主干。搬迁后执行 `flash_next_weights.py refresh` 完整验真生成新路径回执。权重大文件不进Git。5070 Ti preset是显式设备档，不改变24GiB方案和其他机器的全局安全余量。

## 开发验证

构建/源码重放和复用边界见 [build/README.md](build/README.md)，升级与回滚见 [维护规则](docs/maintenance.md)。

```powershell
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

离线验证检查补丁、SHA、wheel源码、native/AOT来源及冻结证据；不启动GPU，也不代替普通wheel、Speed源码、编译载荷、安装器或干净OS验收。旧质量门槛15/20、thinking3/3，间歇长输出标点/token0问题仍未解决；报告不得把它改写成已修复。

历史profile、补丁、合同和材料保留，目录索引见 [integration/speed](integration/speed/README.md)。`q2opt2` 是旧HF主干+Q2专家方案，不能与当前全GGUF混用。早期正文存于 [原README](docs/history/readme-before-review-20261017.txt)，许可见 [NOTICE](NOTICE)。
