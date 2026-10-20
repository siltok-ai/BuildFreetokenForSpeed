# Flash Next GGUF 资源包

当前材料由 [current-release.json](../integration/speed/current-release.json) 选择。Speed 使用 `feat/flash-next`，本仓库使用 `main`。终端用户安装匹配的 Speed 后，只需复制整个资源包，将模型目录指向包根；无需安装本仓库、运行 Python 整理或搬迁后 refresh。

## 来源与目录

主模型唯一来源为 [ISTA-DASLab GSQ-RCO Q2 的完整两片 GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0)。架构配置、248320词表、BPE merges、特殊 token 和模板从 GGUF 导出，不下载 RadixArk 或其他 HF 配套元数据，也不加载 NVFP4/FP8/BF16 主模型。主模型不合并、不重排、不重新量化。

两片 GGUF 没有 MTP 张量。制包者仍从 [官方 Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) 提取31份来源张量，共5214301696字节，转换成823887872字节的紧凑数据，预置在同一个资源包中。接收者无需另外获取或转换 MTP。

```text
flash-next-resources/
  manifest.json                      OFF / ON 两个模型 ID
  checkpoints/
    flash-next-strata-q2/
      gguf/                          两片 GGUF，真实文件
      config.json
      tokenizer.json
      tokenizer_config.json
      generation_config.json
      chat_template.jinja
    flash-next-mtp/                   31份数据与 runtime.json
  speed-config/config.yaml           16GiB 档资源预算
  sources.json
  licenses/
  README.txt                         安装版配置及完整请求
```

配置使用包内相对路径，没有指向制作者机器的目录链接。外层目录可重命名、搬盘或使用带空格的英文路径，保持包内相对层级即可。GGUF 与 MTP 合计约67.25GB；不交付 BF16 转换来源、range-parts 下载缓存或旧 HF 元数据。软件运行库和 AOT 由匹配 Speed 安装包提供。

## 制包者操作

先在独立环境安装当前 profile 对应 FreeToken wheel 及依赖。完整 Speed 打包环境由打包者维护，本仓库只维护 FreeToken/Flash Next 材料。

```powershell
$python = 'D:\envs\speed\Scripts\python.exe'
$release = Get-Content integration/speed/current-release.json -Raw | ConvertFrom-Json
$profile = Get-Content $release.profile -Raw | ConvertFrom-Json
$materials = Join-Path (Get-Location) $release.materials
& $python -m pip install --no-deps (Join-Path $materials ('wheels/' + $profile.wheel_filename))
$inputs = 'D:\flash-next-inputs'
& $python -B tools/flash_next_weights.py download --root $inputs --workers 8
& $python -B tools/flash_next_mtp.py fetch --root $inputs
& $python -B tools/flash_next_mtp.py build --root $inputs
& $python -B tools/flash_next_bundle.py --output D:\flash-next-resources `
  --gguf-dir '<sources下GGUF的固定revision目录>' `
  --mtp-dir "$inputs\checkpoints\flash-next-mtp"
```

`--gguf-dir` 下应有 `Q2_0/<原始两片文件名>`；已有正确文件可直接复用。输出目录须不存在，工具会复制真实文件，需要额外磁盘空间。单独 `flash_next_weights.py prepare` 也生成相同的独立 checkpoint。新格式 `refresh` 只验证，不改写路径或时间。历史来源锁、profile、patch、合同和制品保持原字节。

## 接收者使用

安装包含 FreeToken `0.1.3+speed.strataq2portable1` 的配套 Speed，复制资源包到例如 `D:\models\flash-next`。安装时用 `/MODELSDIR` 指定包根；已安装服务在 `C:\ProgramData\Siltok\Speed\config\operator.override.yaml` 中设置 `service.models_dir`，并合并包内 `speed-config/config.yaml` 的资源预算，保留已有配置。重启服务，按正常流程配置设备许可证和 API 授权。

查询 `/v1/models` 可看到 `flash-next-strata-q2-off` 与 `flash-next-strata-q2-on`。默认推荐 OFF；请求中的 model 选择 ON 即使用 MTP，两入口共享权重。完整文本和 Bonnet 调用说明在包内 README。源码运行时仍需显式设置外部 AOT，组件自检与测试授权边界见 Speed 的 `doc/speed/models/freetoken.md`。

要实际使用MTP草稿加速，请在ON请求中显式设置 `temperature: 0`。当前MTP只处理greedy请求；未传temperature时默认使用采样（no-thinking为0.7，thinking为1.0），此时ON可以正常生成，但沿原生采样路径执行。资源包README的示例已显式设置0；此次只补充说明，未改变采样或MTP行为。

配套 `sm120-16g-portable1` 沿用 perf9 的 RTX5070Ti 16GiB 预算：OFF 永久专家5760、总槽7808；ON 永久专家4992、总槽7040，紧凑 MTP、`paired_cjk` 策略。KV8192、prefill1024、NVML探测、GPU安全余量0.25GiB、commit安全余量4GiB。其他设备需要单独验证，不能据此推断5090性能上限。

## 验证边界

Speed 检查配套版本、依赖、ABI、文件布局及几何，不要求固定 SHA 或制包机时间回执。本仓库保留既有构建来源和材料校验。离线验证执行 `python -B tools/validate_repository.py` 与 `python -B -m unittest discover -s tests -v`，不代替本轮真实 GPU 验证。

GGUF 内嵌模板与旧 HF 模板不同，需要单独验证。删除历史 RadixArk 后的离线搬迁、OFF/ON、中文/SSE/thinking、worker 退出、普通 wheel/保护包消费，以及实际安装、设备许可证、异机测试分别记账，不能继承历史资格。既有间歇长输出问题仍保留。

## 2026-10-20 本机结果

Speed `79c39410`、供应提交 `313e2ab5`，FreeToken `0.1.3+speed.strataq2portable1`，RTX5070 Ti 16GiB。先删除历史 Flash Next RadixArk 配套文件和旧派生 checkpoint，再把资源包移至带空格的新路径、改变 GGUF 文件时间，启用离线模式和空 Hugging Face 缓存。Speed 源码、普通 wheel、embedded Python 保护载荷均通过 OFF/ON 的真实 TCP 同步、中文 SSE、thinking、MTP 所有权和模型进程退出检查。

组件17项及9个子项、Speed相关236项、Build52项通过；3native/27AOT与FT9相同，58组补丁和462个Python文件重放匹配。完整材料及消费记录由当前合同离线复核。首轮保护测试把主进程自带的 conhost 算成模型 worker，修正外置驱动后重测通过；原失败记录保留。此检查点已构建安装器，实际安装和接收者整体验收另行记录。

性能对照双方使用相同 GGUF 元数据、prompt IDs、每模式专家/KV预算、MTP策略和实际512输出token，英文/中文各3次。旧FT9使用其必要的来源包装配置；新版本直接消费资源包。

| 模式/负载 | FT9中位数 tok/s | portable1中位数 tok/s |
| --- | ---: | ---: |
| OFF 英文 | 93.88 | 98.60 |
| OFF 中文 | 99.08 | 99.10 |
| ON 英文 | 104.00 | 104.12 |
| ON 中文 | 105.30 | 104.91 |

本轮未测到 decode 衰退；OFF英文旧组有一次慢值，不能把中位数差异解释为优化收益。主权重和计算内核未改，但GGUF内嵌模板与旧RadixArk模板有差异，上表不验证两种模板的完整质量等价，也不代表5090或长期稳定性验收。全部模型测试使用进程范围测试授权，未验证真实设备许可证。

## 推送后的接收者模拟

在 Speed `79c39410`、Build `b12e265` 推送并核对远端后，用原资源包和同一保护载荷进行独立部署：只用包内Python、运行库和AOT，清除开发环境变量，使用空HF缓存和离线模式；导入路径没有源码或venv目录。按安装器的模板生成独立managed-base，合并包内resource配置，再走正常 `release_profile` 服务入口。授权适配只在外置验收进程替换原生ABI，产品及资源包字节未改。

同一服务三轮OFF/ON切换、第二次启动后的OFF/ON、中文SSE、参数拒绝与正常退出均通过；被替换的模型worker按PID和创建时间核实已退出。OFF/ON各自完整20题均15/20，thinking均3/3，与既有本机协议的五个失败样本一致。此分数不能与5090上另一套20题直接比较。原始报告及配置见当前消费合同的 `receiver` 证据。

未发现额外权重下载、RadixArk引用、MTP文件缺失或本轮引入的加载/切换障碍。接收者需注意：

- 使用包含 `0.1.3+speed.strataq2portable1` 的匹配Speed；旧FT9安装器不支持这个相对路径格式。
- 安装向导的模型目录填写**含manifest.json的包根**，保留包内层级。资源预算在ProgramData的 `config/operator.override.yaml` 合并，随后重启服务。
- 当前档位只在SM120、5070Ti 16GiB和约64GiB主机内存验证；其他容量要检查显存及host commit。它不是5090的性能上限配置。
- `engine_unlicensed` 表示设备授权尚未就绪，不能靠重新下载权重解决；JSON文本可用普通prompt生成，`response_format` 在此模型合同中返回400 `unsupported_feature`。

Windows UAC在实装前返回操作已取消，未创建i09安装目录、服务或ProgramData。本轮是现有系统上的保护载荷接收者模拟，**未取得实际安装/卸载、干净OS或真实设备许可证资格**；此前i08的资格也不转移。验收脚本首轮误写了公开错误码，修正后用新编号完整复验，失败记录保留。
