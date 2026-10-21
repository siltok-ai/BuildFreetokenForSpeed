# Flash Next GGUF 资源包制作与部署

本流程从 Build `main` 生成包含全部模型资源的目录，当前配套 FreeToken `0.1.3+speed.mtpdecode2`。Speed 在功能分支、dev、main 或 release 中均使用本仓库 `main`，并以其 release 合同检查配套关系。接收方取得此目录与匹配的 Speed 安装包后即可使用，无需 Python 转换。

## 获取完整资源包

任意机器均可按下列步骤从固定公开来源下载、转换和整合，不依赖其他机器的目录或已生成资源包。已有按同一流程制作的包时，可直接完整复制后按“接收方部署”使用。

使用相同来源、依赖和制作工具时，模型数据、元数据和部署配置一致；外层目录和文件时间可以不同。无需 FreeToken 源码环境或制包者的本机路径。

## 固定来源

- 主模型：[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0)，修订 `ed59f92082b1e93c0e96d60a8b11aab089b52f09` 的完整两片 Q2 GGUF。
- MTP：[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540)，修订 `de4b8e4d43b917e7706784d8bb445c9af86a3540` 的 31 份 MTP 张量。

配置、248320 词表、BPE merges、特殊 token 和模板从 GGUF 导出，主模型不重新量化。无需 RadixArk 或其他格式的主模型。GGUF 没有 MTP 张量：制包者下载约 5.21GB 的官方 MTP 来源并转换为约 0.824GB 的运行数据，预置到同一个包。

[source-lock.json](../model-weights/qwen3.8-flash-next-gguf-q2/source-lock.json) 固定来源，[mtp-runtime.json](../model-weights/qwen3.8-flash-next-gguf-q2/mtp-runtime.json) 固定转换格式。使用相同仓库修订、依赖版本及这些输入，生成相同的模型资源内容；不要替换模板、修改 preset 或选择其他 GGUF 量化。

## 制包者准备

从 [构建说明](../build/README.md) 的步骤创建 Python 环境并安装固定依赖即可。模型制包工具不需要安装 FreeToken wheel 或 C++/CUDA 编译工具；推理使用的 wheel/AOT 按 [Speed 接入说明](speed-integration.md) 准备。

```powershell
$python = 'D:\envs\ft-build\Scripts\python.exe'
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

MTP 转换输出和清单状态字段见 [MTP 工具说明](../tools/mtp/README.md)。

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
  speed-config/config.yaml           NVML探测、显存及commit安全余量
  sources.json
  licenses/
  README.txt                         接收者操作说明
```

分发整个输出目录。原始 BF16 转换来源、下载缓存、日志、开发环境不随包分发。生成器复制真实文件，运行路径相对于 checkpoint；外层目录可改名、换盘或含英文空格，内部层级保持不变。Speed 安装目录和 AOT 路径也使用英文路径。

`README.txt`、来源说明、manifest 和预算由当前仓库生成；不要手工沿用旧模板。打包应用所需的软件运行库和 AOT 由匹配 Speed 安装包交付，见 [接入说明](speed-integration.md)。

## 接收方部署

1. 安装包含FreeToken `0.1.3+speed.mtpdecode2` 的配套Speed，复制整个资源包到例如 `D:\models\flash-next`。
2. 安装向导的模型目录（`/MODELSDIR`）填写**包含 manifest.json 的包根**。已安装服务在 `C:\ProgramData\Siltok\Speed\config\operator.override.yaml` 设置 `service.models_dir`，合并包内 `speed-config/config.yaml` 的 `resource` 段，保留已有设置。
3. 重启服务，按 Speed 正常流程配置设备许可证和 API 授权。
4. 查询 `GET /v1/models`，选择 `flash-next-strata-q2-off` 或 `flash-next-strata-q2-on`。

```json
{"model":"flash-next-strata-q2-on","messages":[{"role":"user","content":"计算17+25。"}],"max_tokens":128,"temperature":0,"enable_thinking":false,"stream":false}
```

向 `POST http://127.0.0.1:8000/v1/chat/completions` 发送上述 JSON，按实际配置提供 API 授权；`stream=true` 使用 SSE。Bonnet 使用 `/api/openai/v1/chat/completions` 和模型列表中的 `local/` 前缀 ID。

推荐默认 OFF。ON 载入 MTP，推测解码仅用于 greedy 请求，需显式 `temperature=0`；省略时默认使用采样，正常生成但沿原生采样路径执行。两种入口共享主模型文件。

## 设备与使用限制

当前 `sm120-auto` preset在两项入口声明 `engine.resource_profile: auto`，不写死专家池、prefill或准入预算，由Speed已有的硬件档位统一解析。5070 Ti的16GiB档OFF/ON驻留专家5248/4480、总槽7296/6528、GPU预算14.5GiB；32GiB档驻留9216、总槽11264、预算28GiB不变。两档KV8192、动态/staging各1024；ON使用紧凑MTP与 `paired_cjk`，启动限时600秒。建议64GiB主机内存，实际资源仍由正常准入检查决定。

当前生成器已为 OFF/ON 启用 auto，不写入 `moe_q2_resident_experts`、`moe_cache_size`、`max_prefill_length`、`vram_estimate_bytes`、`host_commit_estimate_bytes` 五项固定覆盖值。部署时保留这一配置，并合并包内 NVML/0.25GiB 显存余量/4GiB commit 余量。

`flash_next_speed_config.py --root <资源根> --mode off|on --dry-run` 可预览当前单入口配置；默认选择当前 preset。

资源包内容一致不等于跨 GPU 性能完全一致。首次加载和形状编译需要时间，部署后应检查 OFF/ON、长请求及实际业务输出。结构化文本可通过普通提示生成，当前模型不支持 `response_format`。

`engine_unlicensed` 表示设备授权未就绪，需处理 Speed 授权。离线工具检查和测试授权下的推理不代表真实设备许可证、全新系统或实际安装验收。
