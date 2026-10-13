# Speed dev 接入

consumer-contract.json 记录本分支的来源、材料与验收。Speed 使用同次材料 wheel 和完整 kernel-cache，由构建者保证匹配；Speed 不增加材料 SHA 校验或运行时材料锁。

Speed 文本 worker 关闭视觉/音频编码器，使用 moe_strategy，保持官方 0.1.3 quant_backend 与请求三元组；Windows MoE 指定 expert_load=serial。dense 的 MoE cache 为 0。

examples/historical-nvfp4-manifest.json 提供两模型的相对目录示例；运行时仍需完整本地权重。没有 Flash Next 或 INVALID 条目。
