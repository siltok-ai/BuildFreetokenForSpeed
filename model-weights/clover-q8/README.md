# 在 Speed 安装版中使用 Clover Q8

本包就是本文件所在的目录，包含官方单文件 Q8 GGUF、配置、tokenizer、聊天模板、
`manifest.json` 和 `files.json`。请整目录复制，保留相对位置。GGUF 无需转换。
运行所需的 FreeToken wheel、AOT 和 GGUF 原生模块随匹配的 Speed 安装包提供；
接收机器需要 NVIDIA 驱动，不需要自行安装 MSVC 或 nvcc 来编译模型算子。

接收方须使用包含 Clover 支持的 Speed，例如 `0.2.0+cloverq8.5` 系列安装包，
或已经合入 Clover 支持的 dev 构建。旧版 Speed 不识别 `family: clover`；
仅复制权重目录不会为旧版增加模型支持。

## 放在哪里

模型放在 **Speed 安装向导选择的模型目录 `models_dir`**，不放到程序的
`python/Lib/site-packages` 或 `runtime/kernel-cache`。首次安装的默认模型目录是
`%ProgramData%\Siltok\Speed\models`，也可以在安装时选择其他盘。
静默安装可显式传入 `/MODELSDIR="D:\SpeedModels"`。

已有安装时，以机器当前配置为准。管理员 PowerShell 可只查看路径设置：

```powershell
$ConfigDir = Join-Path $env:ProgramData 'Siltok\Speed\config'
Select-String -Path "$ConfigDir\managed-base.yaml","$ConfigDir\operator.override.yaml" `
  -Pattern '^\s*models_dir:' -ErrorAction SilentlyContinue
```

运维覆盖中的同名设置优先于 managed base。升级会保留运维覆盖；不要仅根据默认
目录猜测实际使用位置。与 Bonnet 配套使用时，两者必须指向同一模型目录。

## 只部署 Clover

可以把本包所在目录直接选为 `models_dir`。例如：

```text
D:\SpeedModels\clover-35b-a3b-q8\  <- 安装时选择这个目录
  manifest.json
  config.json
  Clover-35B-A3B-Preview-Q8_0.gguf
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  generation_config.json
  files.json
  README.md
  ...
```

包内 `manifest.json` 的 `artifacts.weights.path` 为 `"."`，这正好指向本目录。
不必修改 GGUF 文件名、`config.json` 或 tokenizer。

## 添加到已有多模型目录

Speed 读取的是 **`models_dir\manifest.json` 这一份清单**，不会自动递归合并子目录
中的清单。只复制一个子文件夹并不足以注册模型。

假设当前 `models_dir` 是 `D:\SpeedModels`，先把本包复制到
`D:\SpeedModels\clover-35b-a3b-q8`，然后将下面这个对象追加到根清单的 `models` 数组：

```json
{
  "id": "clover-35b-a3b.q8",
  "name": "Clover 35B A3B Q8",
  "type": "language",
  "family": "clover",
  "features": ["chat"],
  "engine": {
    "pipeline": "chat",
    "backend": "FreeTokenBackend",
    "offload": false,
    "resource_profile": "auto"
  },
  "artifacts": {
    "weights": {
      "path": "clover-35b-a3b-q8",
      "layout": "hf_model"
    }
  }
}
```

根清单的顶层格式是 `{"schema":"silitok.models/1","models":[...]}`。
保留已有模型；若这个 ID 已存在，更新对应条目。不要把两个 JSON 文档直接拼接，
也不要把整个根清单替换成只有 Clover 的清单。

路径相对于 **根清单所在目录**。如果目录放在 `language/clover-35b-a3b-q8`，就把上面的
`path` 改为 `"language/clover-35b-a3b-q8"`。这里虽然使用 `hf_model` 目录布局，实际推理
仍读取目录内的 GGUF；不要把 `weights.path` 改成单个 GGUF 文件路径。

子目录里保留原始 `manifest.json` 和 `files.json`，方便单独搬迁和核验本包；
添加到根清单时修改的是根清单，不需要修改包内清单。

## 生效与首次请求

正式服务约每30秒检查当前模型目录的根清单；查询模型列表或请求缺失模型时也会刷新。
修改 `models_dir` 配置后需要重启服务。只修改根清单时可以等待检测，或在管理员
PowerShell 中立即重启：

```powershell
Restart-Service SiltokSpeed
```

先通过服务实际地址查询模型列表。以下示例使用默认本机端口8000；若部署设置了
其他地址或 API 认证，请按机器的现有接入方式调整。

```powershell
$Base = 'http://127.0.0.1:8000'
Invoke-RestMethod "$Base/v1/models"
$Body = @{
  model = 'clover-35b-a3b.q8'
  messages = @(@{ role = 'user'; content = '计算17乘23。只输出整数答案。' })
  enable_thinking = $false
  temperature = 0
  max_tokens = 64
  clover_mode = 'causal'
} | ConvertTo-Json -Depth 5
Invoke-RestMethod "$Base/v1/chat/completions" -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($Body))
```

外部请求使用的是根清单中的 `id`，不是显示名称或文件夹名称。模型按请求加载；
出现在 `/v1/models` 中还不代表已经完成权重加载。

首次加载会做权重加载、图预热和 Triton 形状编译，首个请求可能需要数分钟。
这与现场编译 GGUF 原生扩展不同；后续请求会使用已有缓存。
Speed 正式服务仍需正常的设备授权，本包不提供或绕过许可证。

## 三种解码与容量

| 请求模式 | 支持块宽 | 使用方法 |
| --- | --- | --- |
| `causal` | 不需要指定 | 默认，逐步生成 |
| `selfspec` | 8、16 | 设置 `clover_mode` 和 `clover_width` |
| `denoise` | 8、16、32 | 设置 `clover_mode`、`clover_width`，可选 `clover_threshold` |

块模式要求 greedy，保留 `temperature=0`。`clover_threshold` 默认0.95，取值必须
大于0且小于1。不同模式的延迟和文本表现可能不同；建议先用默认 causal 确认接入。
例如把请求中的模式改为 `"denoise"`，再加上 `clover_width=16`。

`resource_profile="auto"` 按SM120显卡的总显存选择独立档位，单请求运行：
16GiB档（如5070 Ti）为2304个GPU专家槽、KV4096、prefill1024；
总显存至少30GiB时选择32GiB档（如5090），为6144槽、KV8192、prefill2048。
5090不会沿用16GiB档的专家池和上下文容量。
主机建议64GiB内存。两档主机预算均为37.625GiB；16GiB档显存预算14GiB，
Windows可用commit门槛51.625GiB；32GiB档显存预算28GiB，commit门槛65.625GiB。
显存预算另留128MiB，不再额外叠加通用commit余量。物理内存容量不等于可用commit；
其他进程或模型仍会影响准入。请求的输入与最大输出合计需符合对应档位的KV容量。
16GiB档已在RTX 5070 Ti上完整验收；32GiB档在RTX 5090上标定，只做过功能冒烟。
两档锁页权重下限约32.82GiB。128MiB显存余量和0额外commit余量由本模型的资源合同
固定，不采用运维配置中的全局余量；Windows实际可用commit不足时仍拒绝加载。

### 资源自检

可用安装环境执行资源自检。该命令不读取服务的模型清单，默认按16GiB档诊断；
请按实际部署档位明确设置 `$Tier`，`$Weights` 指向模型目录。
独立命令不继承服务的kernel cache设置，下面显式使用同一安装包的材料目录；
若Speed装在其他位置，修改 `$Install`：

```powershell
$Install = 'C:\Program Files\Siltok\Speed'
$Weights = 'D:\SpeedModels\clover-35b-a3b-q8'
$Tier = 'sm120-16g' # 5090等32GiB档改为sm120-32g
$Python = Join-Path $Install 'python\python.exe'
$KernelCache = Join-Path $Install 'runtime\kernel-cache'
& $Python -I -X utf8 -B -c `
  "import json,sys; from silitok_speed.runtimes.freetoken.release import main; raise SystemExit(main(['resource-check','--weights',sys.argv[1],'--extra-json',json.dumps(dict(clover_resource_profile=sys.argv[2],kernel_cache_dir=sys.argv[3]))]))" `
  $Weights $Tier $KernelCache
```

输出区分锁页物理权重下限、主机预算、GPU的Windows commit预算及当前页面文件配置。
只有物理锁页不足时才需要释放物理RAM；commit不足可释放其他提交或扩大页面文件。
自动管理页面文件的未来增长不等于当前额度。若安装版没有`resource-check`命令，
请先升级到包含Clover支持的配套Speed；无需重做GGUF权重。

先保持 `resource_profile="auto"`，不必填写 Flash Next 专用的 Q2 常驻/自适应参数。
手工修改专家池、KV或预填充容量时，需要重新标定并显式提供GPU/主机预算。

## 排查与完整性

- `/v1/models` 没有 Clover：确认当前 `models_dir`，检查它的根清单及相对路径。
- 加载前返回 `resource_insufficient`：查看错误的 `details.reasons`、所需额度和当前
  快照；按显存或commit原因释放资源后重试。不要仅因物理内存空闲就关闭准入检查。
- 提示缺少 AOT 或 GGUF 原生模块：安装匹配的完整 Speed 包；不要用改文件名或
  开启现场原生JIT来绕过材料检查。
- 配置或权重读取失败：先检查是否整目录复制、GGUF长度是否正确，再核验摘要。
- 日志位置以服务配置为准，默认在 `%ProgramData%\Siltok\Speed\logs`。

官方 GGUF 长度为37,863,343,264字节，SHA256为：

```text
834529e7aae000305ae94656782726bcf0ed77976f95a56383447d8903afddab
```

`sources.json` 保存来源锁，`files.json` 保存包内文件的长度与摘要。接收机器有
BuildFreetokenForSpeed 工具环境时，可运行：

```powershell
python tools/clover_bundle.py --verify 'D:\SpeedModels\clover-35b-a3b-q8'
```

仅有 Speed 安装包时，也可使用管理员或普通 PowerShell 对照 `files.json` 核验：

```powershell
$Bundle = 'D:\SpeedModels\clover-35b-a3b-q8'
$Inventory = Get-Content -LiteralPath "$Bundle\files.json" -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($File in $Inventory.PSObject.Properties) {
  $Path = Join-Path $Bundle $File.Name
  if ((Get-Item -LiteralPath $Path).Length -ne $File.Value.size) { throw "长度不符: $($File.Name)" }
  if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $File.Value.sha256) {
    throw "摘要不符: $($File.Name)"
  }
}
```

核验整个 GGUF 会读取约38GB数据，需等待完成。传输完整性通过不等于该机器已经
通过GPU运行验收。许可与来源署名保留在 `LICENSE`、`NOTICE` 和 `sources.json`。
