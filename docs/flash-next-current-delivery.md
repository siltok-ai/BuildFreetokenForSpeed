# 当前 Flash Next 全 GGUF / MTP 交付

唯一选择是 [current-release.json](../integration/speed/current-release.json)：它绑定版本、profile、有序补丁、材料、消费合同与推荐preset。配套 Speed 分支是 `integrate/flash-next-dev-20261018`，Speed的 `silitok_speed/runtimes/freetoken/release.json` 固定实际 wheel SHA 与供应提交。旧 perf8 的真实安装证据保留，新review修订的范围由自己的合同声明，不能继承旧包的安装资格。

本轮 [外部材料边界修复与实际安装验收](flash-next-external-final-20261019.txt) 删除Speed旧qwen36包及全部AOT DLL；新源码、普通wheel、迁移保护载荷和实际i08安装均独立验收，测试安装已卸载并完成15项清理。既有MTP下降/长输出问题保留，非干净OS或异机验证。

此前Speed输入边界与整体diff审查见 [本轮review](flash-next-blast-review-20261018.txt)；性能改动、相同材料及此前迁移打包验证见 [perf9说明](flash-next-perf9-20261018.txt)；此前包装层和MTP别名兼容记录见 [runtime2说明](flash-next-runtime2-20261018.txt)。公共`models[].id`和旧`Qwen36Backend`继续有效，`FreeTokenBackend`是等价新入口。

## 环境与匹配材料

当前合同：Windows x64、CPython3.12、Torch2.11.0+cu130、triton-windows3.6.0.post26、apache-tvm-ffi0.1.13.post3、SM120。系统其他项目的Triton版本不能替代这套合同。Torch的cu130标记与构建Toolkit版本分别记录。

```powershell
# Build仓库根目录；先在独立venv准备Speed完整运行依赖。
$release = Get-Content integration/speed/current-release.json -Raw | ConvertFrom-Json
$profile = Get-Content $release.profile -Raw | ConvertFrom-Json
$materials = Join-Path (Get-Location) $release.materials
$python = 'D:\envs\speed\Scripts\python.exe'
& $python -m pip install --no-deps (Join-Path $materials ('wheels/' + $profile.wheel_filename))
& $python -B tools/verify_current_release.py
```

wheel内3个native模块与配套27个AOT DLL成对使用。构建期校验完整字节，worker检查版本、ABI、GPU和必要文件；不要在每个生成步骤重复扫描DLL。`FREETOKEN_DISABLE_JIT=1`禁止原生扩展临时构建，不禁止Triton编译。新二进制或新源码使用新版本及新输出目录，已接受材料不覆盖。

## 准备主模型和可选预测层

```powershell
$models = 'D:\models\flash-next'
& $python -B tools/flash_next_weights.py plan --root $models
& $python -B tools/flash_next_weights.py download --root $models --workers 8
& $python -B tools/flash_next_weights.py prepare --root $models
& $python -B tools/flash_next_weights.py verify --root $models

# 仅开启MTP时需要：固定revision的31个BF16来源张量及独立转换目录。
& $python -B tools/flash_next_mtp.py fetch --root $models
& $python -B tools/flash_next_mtp.py build --root $models
& $python -B tools/flash_next_mtp.py verify --root $models
```

主模型是 ISTA-DASLab 固定 GSQ-RCO Q2 revision 的完整两片GGUF，大小37623740192与28800138432字节。五份HF文件只提供配置、tokenizer和模板，不加载RadixArk HF主干。原始Q2专家、分段量化主干、完整主词表和IQ4_NL PLE均保留。

已有正确GGUF可用 `download --metadata-only`，再 `prepare --gguf-dir <固定revision目录>`；其中包含 `Q2_0/<两片文件名>`。来源与checkpoint同盘以保留可搬迁相对描述；工具不移动、覆盖或复制已有权重。

MTP从固定Qwen BF16 revision提取31份张量，共5214301696字节，再转换为823887872字节的FreeToken数据；不下载360GB主模型，不用Strata S2文件替代。发布manifest SHA保持 `610c51dadbfa1b38775a88da708270c0084c84214d934d9192542267f68dc81d`，历史 `packed_unvalidated_runtime` 字段不表示当前验收失败。实际路径、转换耗时和校验时间使用本次回执。

GGUF引擎读取绝对运行路径，来源描述另保留相对位置。搬迁后对新根执行 `flash_next_weights.py refresh --root <新根>` 和 `verify`，重新完整SHA核验并生成本机mtime回执；不要手改时间绕过校验。

## 5070 Ti 配置

```powershell
$deployment = 'D:\deploy\flash-next-off'
& $python -B tools/flash_next_speed_config.py --root $models --deployment-root $deployment --mode off --preset sm120-16g-perf9
$env:SILITOK_CONFIG = Join-Path $deployment 'speed-config'
$env:SILITOK_FREETOKEN_KERNEL_CACHE_DIR = Join-Path $materials 'kernel-cache'
# 在配套Speed仓库或已安装Speed环境运行；正常设备许可证仍然生效。
& $python -m silitok_speed.cli serve --models-dir $deployment --gpu-resource-probe nvml --port 8011
```

工具通过目录联接/符号链接复用共享checkpoint，不复制权重，拒绝覆盖已有部署文件。`on`和`off-matched`分别使用新的部署目录。开启模式设置独立 `artifacts.mtp/layout=freetoken_mtp`，不在extra中直接放路径。

| 模式 | 永久专家 | 动态 / prefill槽 | 总槽 |
| --- | ---: | ---: | ---: |
| off | 5760 | 1024 / 1024 | 7808 |
| on、off-matched | 4992 | 1024 / 1024 | 7040 |

两模式KV8192、prefill1024、额外前缀快照一槽、完整专家和主词表。preset显式启用mapped embedding、有界PLE行缓存、热度衰减及受控prefill工作区借用；on使用紧凑MTP与`paired_cjk`。`full`策略仍可选，非greedy沿原采样路径。

本机显存估计15GiB、host heap估计31GiB；Windows有效commit包含显存映射，两项不能混为一项。NVML不在主进程创建CUDA上下文，GPU安全余量0.25GiB、commit安全余量4GiB；内部rebuild额外预留显式0不等于关闭资源监督。该配置接近16GiB显存上界，不保证其他机器或并行GPU负载可用。历史24GiB、FT2余量档和各perf候选preset保留，但必须配对其原profile；它们的默认容量或旧性能不代表当前修订。

## 安装包与验证

Speed源码仓库和普通wheel不携带FreeToken AOT DLL；当前及历史二进制材料由BuildFreetokenForSpeed维护。源码和普通wheel显式配置外部缓存，Windows安装器将打包者提供的输入复制为 `app/runtime/kernel-cache`，WinSW和launch.cmd选择该目录。安装版的资源覆盖项合并到 `C:\ProgramData\Siltok\Speed\config\operator.override.yaml`，保留用户其他设置，模型目录使用 `/MODELSDIR`。缓存路径使用ASCII；TVM-FFI在中文路径加载AOT受限。

```powershell
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

普通wheel组件/整模型、Speed源码、编译载荷、实际安装和干净OS分别验收。迭代优先前几层；安装器/服务改动或重要发布检查点再做实际安装与归属清理。性能比较固定权重、输入、实际512输出、模式和资源配置，分别报告TTFT、prefix冷热与decode。完整方法、改前/改后记录及review结论见 [晚间review](flash-next-review2-20261017.txt)及[下午review](flash-next-review-20261017.txt)；历史perf8实装数据见 [原交付](flash-next-5070ti-perf8.txt)。

保留间歇长输出标点/token0已知问题，旧质量门槛15/20、thinking3/3；不宣称已修复、长期压力或干净OS通过。冻结原始报告和逐文件SHA是验收依据，离线验证不会重新运行GPU，也不能把旧安装结果写成新代码通过。
