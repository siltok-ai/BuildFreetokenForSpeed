# Speed dev 接入

consumer-contract.json 是唯一当前消费合同。Speed 必须使用同次材料 wheel 和完整 kernel-cache，绑定实际 SHA，并在正式打包时核对。

Speed 文本 worker 关闭视觉/音频编码器，使用 moe_strategy，保持官方 0.1.3 quant_backend 与请求三元组；Windows MoE 指定 expert_load=serial。dense 的 MoE cache 为 0。

examples/historical-nvfp4-manifest.json 提供两模型的相对目录示例；运行时仍需完整本地权重。没有 Flash Next 或 INVALID 条目。
