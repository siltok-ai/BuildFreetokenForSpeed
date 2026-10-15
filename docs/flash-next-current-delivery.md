# 当前 Flash Next 全 GGUF / MTP 材料交付

当前选择在 `integration/speed/current-release.json`。FreeToken引擎为 `0.1.3+speed.strataq2mtp2`，默认与Speed分支 `integrate/flash-next-strata-q2-mtp-5070ti` 的 `f13ffb99` 配套，直接基于dev `fc7dd6a7`；探索 `3e251687` 及其合同保留。5070 Ti适配留在Speed，保持引擎wheel/AOT、量化数学、31份MTP数据及Speed批准的manifest SHA不变。当前合同绑定最终整合提交f13ffb99，材料供应固定1d9ccc0；原探索合同绑定运行e0524007/交付3e251687；完整安装/卸载已验证，带已知长输出问题接受，详情见[本机验收](flash-next-5070ti-acceptance.md)。旧24GiB合同及旧dev材料分支保持原样。

## 环境和制品

Windows x64、CPython3.12、Torch2.11.0+cu130、apache-tvm-ffi0.1.13.post3、triton-windows3.6.0.post26、SM120是当前已验收组合。先按 `build/README.md` 在新venv安装冻结依赖；FreeToken构建依赖锁不等于Speed完整运行环境，不宣称无网络或任意GPU可用。

在仓库根目录执行：

```powershell
$release = Get-Content integration/speed/current-release.json -Raw | ConvertFrom-Json
$materials = Join-Path (Get-Location) $release.materials
$python = 'D:\envs\speed\Scripts\python.exe'
& $python -m pip install (Join-Path $materials 'wheels/freetoken-0.1.3+speed.strataq2mtp2-cp312-cp312-win_amd64.whl')
& $python -B tools/verify_current_release.py
```

wheel自带3个native模块，配套cache含27个AOT DLL。始终成对选择；旧`q2opt2`为HF主干/Q2专家混合权重，不能当完整GGUF入口。重新编译用当前默认profile，输出新目录；旧profile仍可显式选择。不修改或重命名旧wheel来冒充新版本。

## 准备主模型

```powershell
$models = 'D:\models\flash-next'
& $python -B tools/flash_next_weights.py plan --root $models
& $python -B tools/flash_next_weights.py download --root $models --workers 8
& $python -B tools/flash_next_weights.py prepare --root $models
& $python -B tools/flash_next_weights.py verify --root $models
```

`plan`不访问网络/创建目录。下载只读固定revision，完整SHA核验两片GGUF（37,623,740,192与28,800,138,432字节），不会将只下载第一片当完整模型。已有正确文件复用，已有坏文件拒绝，不覆盖。五份HF文件仅为配置、tokenizer和聊天模板；不下载或加载HF主干权重。

已有完整GGUF可先只下载元数据，然后显式复用原目录：

```powershell
& $python -B tools/flash_next_weights.py download --root $models --metadata-only
& $python -B tools/flash_next_weights.py prepare --root $models --gguf-dir 'D:\verified\gguf-revision-root'
```

`--gguf-dir`中须有`Q2_0/<两片文件名>`。为保证相对来源描述能搬迁，来源与checkpoint必须同盘；跨盘复用应先在目标盘自行放置一份正确文件。工具不会移动、删除或原地修改现有模型，也不依赖本机HardLink才正确。使用自定义外部来源时，搬迁应携带对应来源树或在`refresh`中显式指定新来源目录。

FT2引擎要求绝对GGUF路径。因此运行用`config.json`包含本机绝对路径及当前mtime验真回执，稳定的`strata-q2-preparation.json`另外保存相对来源位置。目录整体搬迁后：

```powershell
& $python -B tools/flash_next_weights.py refresh --root 'F:\models\flash-next'
& $python -B tools/flash_next_weights.py verify --root 'F:\models\flash-next'
```

refresh重新完整SHA校验后才生成目标机器回执，不通过修改权重时间来绕过校验。只处理本工具持有、模型配置未被意外修改的checkpoint。旧golden目录不会被收编或覆盖。

## 可选 MTP

```powershell
& $python -B tools/flash_next_mtp.py plan --root $models
& $python -B tools/flash_next_mtp.py fetch --root $models
& $python -B tools/flash_next_mtp.py build --root $models
& $python -B tools/flash_next_mtp.py verify --root $models
```

来源是Qwen固定BF16 revision的31个MTP张量，合计5,214,301,696字节，通过严格206范围请求提取；不下载360GB主模型，不回退main。输出为FreeToken原始Q2专家布局与Q8/BF16/FP32小参数，共823,887,872字节；Strata S2文件不是替代品。

构建逐项核对31个张量名称、形状、类型、大小、来源SHA和派生SHA，全部数据正确才安装本仓库`model-weights/qwen3.8-flash-next-strata-q2/mtp-runtime.json`的原始字节，SHA仍为`610c51...`，因此当前Speed无需换引擎版本或削弱哈希核验。该不可变manifest保留首次批准转换的历史字段；它们不表示本次耗时。本机路径、校验时间、实际转换耗时放在来源verification和新`conversion-report.json`中，不重新生成发布身份。数据不一致或转换中断时不会发布可加载的runtime.json。

MTP数据和主模型在不同checkpoint目录。关闭MTP不需要下载预测层。

## 配置 Speed 与复现实测预算

显式生成一套模式，默认`off`；对照包括`off`（原最佳缓存）、`on`和`off-matched`（与MTP相同专家预算）：

```powershell
& $python -B tools/flash_next_speed_config.py --root $models --mode off --preset sm120-24g-measured --dry-run
& $python -B tools/flash_next_speed_config.py --root $models --mode off --preset sm120-24g-measured
$env:SILITOK_CONFIG = Join-Path $models 'speed-config'
$env:SILITOK_FREETOKEN_KERNEL_CACHE_DIR = Join-Path $materials 'kernel-cache'
& $python -m silitok_speed.cli serve --models-dir $models --port 8011
```

生成模型根的`manifest.json`、`speed-config/config.yaml`和`speed-inputs.json`。已有部署文件拒绝覆盖；要换模式，保留旧配置并在新模型根准备对应模式，不修改机器全局配置。`on`显式使用`artifacts.mtp/layout=freetoken_mtp`，不在engine.extra中放路径。

这个preset是已测24GiB/SM120单机实验：NVML资源探测，GPU安全64MiB/commit安全2GiB。off为永久9216/总槽11264、估计GPU22.625GiB/host24.875GiB；on为7168/9216、GPU21.75GiB/host27GiB；off-matched为7168/9216、GPU21GiB/host27GiB。KV8192、动态槽1024、disk PLE、radix、graph1、max_prefill_length4096。必须结合实际整机commit、其他GPU进程及硬件重新评估，不能当通用默认或保证100tps。

RTX 5070 Ti 16GiB、约64GB RAM的交接配置另有两个显式preset；它们要求配套Speed已支持较小驻留池和独立GPU/host预算，原24GiB配置保持原样：

| preset | off永久/总槽 | on与off-matched永久/总槽 | GPU安全余量 | commit安全余量 |
| --- | --- | --- | --- | --- |
| `sm120-16g-measured` | 3584/5632 | 2304/4352 | 256MiB | 4GiB |
| `sm120-16g-roomy-measured` | 2816/4864 | 1536/3584 | 1GiB | 4GiB |

前者为接近显存上界的单worker方案，需要足够空闲显存，不能与其他GPU负载同时保证可用；后者保留更大显存余量。原生和Speed backend结果表明两档均能运行完整模型，MTP默认仍关闭。实际HTTP、编译安装包与后续整合分支的验收状态以消费合同及新机记录为准，不能将旧24GiB证据当成本机通过。首次遇到新的短prefill形状时，Triton可能编译新的`_ensure`等内核；`FREETOKEN_DISABLE_JIT=1`关闭的是原生扩展临时构建，不关闭Triton JIT。记录首次编译、TTFT与稳态decode时须区分这几个阶段。

切换模式不需要复制权重。`--root`指向已经准备和校验的共享模型根，`--deployment-root`使用新的独立配置目录；工具在Windows用标准库创建checkpoint目录联接，在其他系统创建目录符号链接，保持manifest里的artifact为相对路径。已有配置或checkpoint入口会拒绝覆盖：

```powershell
$deployment = 'C:\workspace\cfg\siltok-speed\flash-accept\delivery-off'
& $python -B tools/flash_next_speed_config.py --root $models --deployment-root $deployment --mode off --preset sm120-16g-measured
$env:SILITOK_CONFIG = Join-Path $deployment 'speed-config'
$env:SILITOK_FREETOKEN_KERNEL_CACHE_DIR = Join-Path $materials 'kernel-cache'
& $python -m silitok_speed.cli serve --models-dir $deployment --gpu-resource-probe nvml --port 8011
```

`on`与`off-matched`分别生成到另外两个新部署目录。搬迁共享权重后，先对新的权重根执行`refresh`及`verify`，再生成新的部署目录和联接；不会自动改写旧配置。这里的配置生成不代表其他OS/GPU已有推理资格。

避免继承旧`SILITOK_FREETOKEN_RUNTIME_DIR`或旧cache环境变量。开发源码可使用包内cache，但发布/验证应显式指向本release。Triton缓存可重新生成，首次编译/预热、TTFT和稳态decode分开记录。

## 安装版的设备资源配置

配置生成工具输出的是直接CLI使用的Speed配置。安装器则通过`release_profile`合并`managed-base.yaml`与`operator.override.yaml`：安装时用`/MODELSDIR=<部署根>`选择生成的manifest目录，设备资源预算放在`C:\ProgramData\Siltok\Speed\config\operator.override.yaml`。例如16GiB较快档：

```yaml
resource:
  gpu_probe: nvml
  vram_safety_gb: 0.25
  commit_safety_gb: 4
```

已有override应合并相关字段并保留其他设置，不能覆盖已有用户配置或修改managed-base假装全局默认改变。roomy档的GPU余量改为1GiB。修改后重启服务。安装包默认通过WinSW选择包内kernel-cache，不应沿用开发机的外部cache路径。

release override的passthrough有白名单，不能将`TRITON_CACHE_DIR`等任意环境变量写进去。需要单独隔离JIT缓存时，在WinSW支持的`<env name="TRITON_CACHE_DIR" value="..." />`等环境项中配置，保留已批准的release字段校验。本机安装验收会记录这类部署配置及冷编译耗时；安装成功本身不等于模型已经验收。

## RTX 5070 Ti 实测

本机源码真实HTTP/SSE已测：MTP关闭、永久3584专家时，70-token与4096-token输入的512-token decode中位分别为71.76/68.94 tps；MTP开启2304专家为61.04/62.64，同容量关闭为66.56/63.37。每组各3次实际512输出。同容量六组输出一致，本机保持MTP默认关闭。较快off方案最低空闲显存约0.487GiB，需留意其他GPU负载；另提供余量更大的roomy档。完整环境、TTFT、统计口径、质量范围、原始证据及尚待完成的安装包验收见 [5070 Ti交接实测](flash-next-5070ti-acceptance.md)。这些数字不替代历史5090 D v2记录，也不作为其他机器的性能保证。

新增本机源码证据可离线复核：

```powershell
python -B tools/verify_flash_next_5070ti.py
```

它核对封存字节、原始SSE速率/输出及正式CLI三模式，不启动GPU，也不把源码结果升级为整包或整合分支通过。

## 验收范围及后续消费端工作

本仓库默认指针/完整来源工具的验证与历史本机推理验收分开保存。原20题质量/性能、旧27B/35B、MTP事务/取消等证据仍绑定原engine/wheel。新工具要通过真实两片SHA、异路径MTP全转换、移动目录后的刷新、当前Speed实际加载和无开发路径干扰的导出验证。

Speed自己的源码行尾/包清单复核、新Cython单DLL整包、安装器与干净机验收仍属于Speed后续范围；本次不把它们写成通过。完整离线wheelhouse和其他OS/GPU的profile同样尚未交付。
