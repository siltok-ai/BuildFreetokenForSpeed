# Flash Next GGUF 资源包制作与部署

本流程生成一个包含全部模型资源的目录，配套 Speed 的 `feat/flash-next` 和 FreeToken `0.1.3+speed.strataq2portable1`。接收方取得这个目录与匹配的 Speed 安装包后即可使用，无需 Python 转换。

## 优先使用现成资源包

优先从 **siltok31** 获取已处理好的 `C:\workspace\flash-next-resources`，完整复制整个目录，然后按本文的“接收方部署”使用。包内主模型、配置、tokenizer、模板、MTP 和资源预算已配套，无需再次处理。

不方便获取或希望从头制作时，使用下列固定来源、依赖和制作步骤。自行制作与现成包使用同一输入和转换格式，生成的模型资源内容一致；外层目录和文件时间可以不同。

## 固定来源

- 主模型：[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0)，修订 `ed59f92082b1e93c0e96d60a8b11aab089b52f09` 的完整两片 Q2 GGUF。
- MTP：[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540)，修订 `de4b8e4d43b917e7706784d8bb445c9af86a3540` 的 31 份 MTP 张量。

配置、248320 词表、BPE merges、特殊 token 和模板从 GGUF 导出，主模型不重新量化。无需 RadixArk 或其他格式的主模型。GGUF 没有 MTP 张量：制包者下载约 5.21GB 的官方 MTP 来源并转换为约 0.824GB 的运行数据，预置到同一个包。

[source-lock.json](../model-weights/qwen3.8-flash-next-gguf-q2/source-lock.json) 固定来源，[mtp-runtime.json](../model-weights/qwen3.8-flash-next-gguf-q2/mtp-runtime.json) 固定转换格式。使用相同仓库修订、依赖版本及这些输入，生成相同的模型资源内容；不要替换模板、修改 preset 或选择其他 GGUF 量化。

## 制包者准备

从 [构建说明](../build/README.md) 的步骤创建环境并安装固定依赖即可，不制作 wheel/AOT 时无需安装 C++/CUDA 编译工具。安装 [当前材料](../artifacts/README.md) 中的 FreeToken wheel，供配置导出使用：

```powershell
$python = 'D:\envs\ft-build\Scripts\python.exe'
$release = Get-Content integration/speed/current-release.json -Raw | ConvertFrom-Json
$profile = Get-Content $release.profile -Raw | ConvertFrom-Json
$materials = Join-Path (Get-Location) $release.materials
& $python -m pip install --no-deps (Join-Path $materials ('wheels/' + $profile.wheel_filename))
$inputs = 'D:\flash-next-inputs'
$plan = & $python -B tools/flash_next_weights.py plan --root $inputs | ConvertFrom-Json
```

下载和转换使用仓库外的短英文路径。最终资源包约 67.3GB。

```powershell
& $python -B tools/flash_next_weights.py download --root $inputs --workers 8
& $python -B tools/flash_next_mtp.py fetch --root $inputs
& $python -B tools/flash_next_mtp.py build --root $inputs
& $python -B tools/flash_next_mtp.py verify --root $inputs
& $python -B tools/flash_next_bundle.py --output D:\flash-next-resources `
  --gguf-dir $plan.paths.gguf --mtp-dir $plan.paths.mtp
```

GGUF 下载可在失败后重试同一命令，保留 `range-parts` 缓存以续传。已有两片正确文件时，将它们置于 `$plan.paths.gguf\Q2_0\`，保持原文件名，下载器会核验后复用。输出资源包必须是新目录；转换中断后保留诊断，将不完整的 MTP 输出移到另一个位置再重建，不与现用数据混合。

已有配套的紧凑 MTP 时可直接将 `--mtp-dir` 指向它，跳过 fetch/build；制包工具仍检查其清单和数据。直接制作整合包不需要先运行 `flash_next_weights.py prepare`，避免多复制一份主模型。

## 输出与分发

```text
flash-next-resources/
  manifest.json                      OFF / ON 两个模型入口
  checkpoints/
    flash-next-strata-q2/
      gguf/                          两片完整 GGUF，真实文件
      config.json
      tokenizer.json
      tokenizer_config.json
      generation_config.json
      chat_template.jinja
      gguf-preparation.json
    flash-next-mtp/                   runtime.json 与 31 个数据文件
  speed-config/config.yaml           默认 16GiB 档资源预算
  sources.json
  licenses/
  README.txt                         接收者操作说明
```

分发整个输出目录。原始 BF16 转换来源、下载缓存、日志、开发环境不随包分发。生成器复制真实文件，运行路径相对于 checkpoint；外层目录可改名、换盘或含英文空格，内部层级保持不变。Speed 安装目录和 AOT 路径也使用英文路径。

`README.txt`、来源说明、manifest 和预算由当前仓库生成；不要手工沿用旧模板。打包应用所需的软件运行库和 AOT 由匹配 Speed 安装包交付，见 [接入说明](speed-integration.md)。

## 接收方部署

1. 安装包含 FreeToken `0.1.3+speed.strataq2portable1` 的配套 Speed，复制整个资源包到例如 `D:\models\flash-next`。
2. 安装向导的模型目录（`/MODELSDIR`）填写**包含 manifest.json 的包根**。已安装服务在 `C:\ProgramData\Siltok\Speed\config\operator.override.yaml` 设置 `service.models_dir`，合并包内 `speed-config/config.yaml` 的 `resource` 段，保留已有设置。
3. 重启服务，按 Speed 正常流程配置设备许可证和 API 授权。
4. 查询 `GET /v1/models`，选择 `flash-next-strata-q2-off` 或 `flash-next-strata-q2-on`。

```json
{"model":"flash-next-strata-q2-on","messages":[{"role":"user","content":"计算17+25。"}],"max_tokens":128,"temperature":0,"enable_thinking":false,"stream":false}
```

向 `POST http://127.0.0.1:8000/v1/chat/completions` 发送上述 JSON，按实际配置提供 API 授权；`stream=true` 使用 SSE。Bonnet 使用 `/api/openai/v1/chat/completions` 和模型列表中的 `local/` 前缀 ID。

推荐默认 OFF。ON 载入 MTP，推测解码仅用于 greedy 请求，需显式 `temperature=0`；省略时默认使用采样，正常生成但沿原生采样路径执行。两种入口共享主模型文件。

## 设备与使用限制

当前 preset 在 RTX 5070 Ti 16GiB、约 64GiB 主机内存验证：OFF 永久专家 5760/总槽 7808；ON 永久专家 4992/总槽 7040，紧凑 MTP 和 `paired_cjk`。KV 8192、prefill 1024、启动限时 600 秒。其他设备按显存及 host commit 调整预算，这不是 32GiB 显卡的性能上限配置。

资源包内容一致不等于跨 GPU 性能完全一致。首次加载和形状编译需要时间；间歇长输出标点/token0 异常仍是已知限制。结构化文本可通过普通提示生成，当前模型不支持 `response_format`。

`engine_unlicensed` 表示设备授权未就绪，需处理 Speed 授权。离线工具检查和测试授权下的推理不代表真实设备许可证、全新系统或实际安装验收。
