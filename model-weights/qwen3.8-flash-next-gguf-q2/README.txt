Flash Next GGUF Q2 资源包

本包包含 Flash Next 推理所需的全部模型资源：原始两片 GSQ-RCO Q2 GGUF、
从 GGUF 生成的配置/tokenizer/chat template，以及已转换好的官方 Qwen MTP。
主模型与配套元数据均来自 ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF。
MTP 单独来自官方 Qwen/Qwen3.8-Flash-Next；它不在原始 GGUF 中。
不需要 RadixArk、NVFP4 主模型、BF16 主模型或下载/转换临时缓存。

准备：安装包含 FreeToken 0.1.3+speed.strataq2portable1 的配套 Speed。
软件运行库和 AOT 由 Speed 安装包提供；本包不代替 Speed 安装器或设备许可证。
当前预算适用于 Windows x64、SM120、RTX5070Ti 16GiB，建议64GiB主机内存。
模型准入预算为15GiB显存、31GiB host heap，另留4GiB系统commit余量。
其他设备需要按显存及 host commit 调整预算，不应把此档位当作性能上限。

使用：
1. 把整个文件夹解压或复制到自选位置，例如 D:\models\flash-next。
   保持包内相对层级。可以重命名外层目录、换盘、使用带空格的英文路径。
   不需要运行 Python 整理、转换或 refresh，也不需要联网。
2. 安装 Speed 时，将模型目录 /MODELSDIR 设为这个文件夹的绝对路径。
   已安装的服务在其 ProgramData 配置目录的 operator.override.yaml 中设置：
     service:
       models_dir: 'D:/models/flash-next'
   将本包 speed-config/config.yaml 的 resource 段合并到该 override，保留已有配置。
   配置变更后重启 Speed。设备许可证按正常流程配置。
3. GET http://127.0.0.1:8000/v1/models，确认能看到两个模型 ID：
     flash-next-strata-q2-off   关闭 MTP，推荐默认使用
     flash-next-strata-q2-on    开启 MTP
   两个入口共享同一份权重；Speed 按现有资源调度规则切换。
4. POST http://127.0.0.1:8000/v1/chat/completions，JSON 示例：
   {"model":"flash-next-strata-q2-off","messages":[{"role":"user","content":"计算17+25。"}],"max_tokens":128,"temperature":0,"enable_thinking":false,"stream":false}
   将 model 改成 flash-next-strata-q2-on 即使用 MTP。stream=true 可接收 SSE。
   当前 MTP 草稿加速仅用于 greedy 请求，请显式设置 temperature=0。
   省略 temperature 时默认使用采样；ON 仍能生成文本，但采样请求沿原生路径执行。
   端口及 API 授权按实际服务配置填写；通过 Bonnet 调用时使用列表中 local/ 前缀的 ID。

包内 checkpoints、manifest.json 和 sources.json 一起保留。
sources.json 记录来源和兼容版本；licenses 保存相关许可文本。
首次加载和首次形状编译需要时间；模型启动限时为600秒。
