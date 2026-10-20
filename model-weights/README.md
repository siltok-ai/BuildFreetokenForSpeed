# Flash Next 模型资源

当前使用 [qwen3.8-flash-next-gguf-q2](qwen3.8-flash-next-gguf-q2/)：

优先完整复制 **siltok31** 的 `C:\workspace\flash-next-resources` 现成资源包。不方便获取或希望自行制作时，按 [制作与部署说明](../docs/flash-next-gguf-bundle.md) 生成内容相同的包。

- 主模型及配套配置、tokenizer、模板：ISTA-DASLab 固定修订的完整两片 GSQ-RCO Q2 GGUF。
- MTP：官方 Qwen 固定修订的 31 份来源张量，制包时转换并预置到同一资源包。
- 运行时无需 RadixArk、其他格式的主模型或下载临时缓存。

[制作与部署说明](../docs/flash-next-gguf-bundle.md) 提供完整命令。目录内 `source-lock.json`、`mtp-runtime.json` 和许可证是制包输入；`README.txt` 随包分发给接收方。权重、转换来源和最终资源包放在仓库外。
