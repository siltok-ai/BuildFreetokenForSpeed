# 双模型 NVFP4 验收

`acceptance.json` 汇总最终二进制身份、最终 Speed 提交、210 个模块源码 SHA 一致性、
性能、容量及完整输出 SHA。编译清单保留压缩历史前的 commit；最终 PR 的代码由模块
SHA 对照绑定，不伪造二进制的 git 身份。

- `simple-packaged-*.json`：删除新增材料 SHA 校验后的最终正式包，两模型加载及生成。
- `minimal-packaged-*.json`：此前正式包的完整功能/性能回归，含真实 TCP HTTP；最终改动后复用该记录。
- `reviewed-packaged-*.json`：此前完整正式包回归；`source-*.json`：源码运行输出。
- `direct-freetoken-qwen38.json` / `speed-comparison-qwen38.json`：相同输入及采样的独立对照。
- `weights-identity.json`：本机权重路径、来源、revision 证据和实际文件 SHA。
- `*-capacity.json`：100 ms 采样的整机 GPU、系统 commit 和本次子进程树容量。
- `relocated-ascii-*.json`：独立复制包在英文含空格目录运行；
  `relocated-qwen36.log`：中文目录下 TVM-FFI DLL 加载失败的原始输出。
- `performance-investigation.md`：公开社区一手信息及性能比较边界。

记录里的绝对路径是测试地点，不作为输入默认值。复现脚本通过参数接收其他机器的路径。
生产者 `build-record.json` 保留材料构建时的历史状态；后续 GPU 验收以本目录及消费合同为准。
完整性检查保证材料身份，不宣称其他 GPU、安装器或干净机器已通过。

## 复现

准备固定 CPython 3.12 / Torch 2.11.0+cu130 环境，安装本分支 wheel，并提供 19 DLL cache。
`$Weights`、`$Cache`、`$SpeedSource`、`$PackagePython` 和 `$Out` 均由执行者填写。
独立 FreeToken 对照不导入 Speed，复现脚本只把原始测量脚本的固定输入路径改为参数：

```powershell
python -B .\direct-freetoken-bench.py --model $Weights --kernel-cache $Cache --output $Out
python -B .\speed-comparison-bench.py --source $SpeedSource --model $Weights --kernel-cache $Cache --output $Out
# 正式编译包后端；profile 可为 qwen36 或 qwen38。
& $PackagePython -I -B .\run-validation.py --packaged --mode backend --source $SpeedSource `
  --model $Weights --quant nvfp4 --profile qwen38 --kernel-cache $Cache `
  --repeats 3 --tokens 128 --long-prompt-tokens 2048 --long-repeats 3 --quality --output $Out
```

`run-validation.py --mode http --tcp` 会启动真实 Uvicorn 随机本地端口，并在该外部验收
进程临时绕过许可证入口。这是本次用户授权的测试措施；正式包仍需设备证书。
Qwen3.8 的 HTTP smoke 额外使用 `--vram-safety-gib 0.5`，未修改产品默认 2 GiB 余量。
显存较紧，应结合实际桌面占用和负载配置准入；此次为单模型单请求。

两模型覆盖生成、thinking、同步/SSE、取消后复用及卸载。中文案例在 256 token 上限
截断；JSON 文本生成不意味着 schema 约束已支持。只验收本地 NVFP4 权重，未验收
FP8、视觉、MTP、256K 实际 KV、安装器或干净机器。

## 改动 review

官方源码冻结为 0781324，三个补丁只修改 12 个文件，保留官方推理/量化实现。
Windows 平台适配、19 个必需 AOT DLL 和包版本构成材料；没有 Flash Next/Q2/RowStore 改造。
Speed 使用官方 `moe_strategy`、关闭非文本编码器、Windows MoE 串行读取，保留官方
`quant_backend` 调用。删除 worker 全局 POSIX shim 与生产者重复 wheel 哈希。
按用户要求，Speed 最终提交删除新增材料 SHA 校验和材料锁，文档指向本分支，
由构建者保证 wheel/cache 一致。生产者保存来源和字节身份用于记录，未添加 Speed 运行时校验。
删除校验后只重测正式打包、两模型加载和生成，之前完整回归的功能与性能记录保留。
