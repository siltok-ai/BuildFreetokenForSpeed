# 模型权重来源与分发

本目录由公司仓库 `siltok-ai/BuildFreetokenForSpeed` 维护，说明 Speed 实际消费的模型文件、来源身份、组合方式和分发流程。大权重存放在模型托管服务或对象存储中；Git 只保存说明、来源锁、校验清单和工具。

- [Qwen3.8-Flash-Next：BF16 主干 / FP8 PLE / 完整 Q2_0 专家](qwen3.8-flash-next-q2/README.md)

引擎 wheel/cache 与模型权重是不同交付物。修改权重布局、精度或来源后需要新的模型版本与真实推理验收；补充说明不会改变已冻结的引擎制品。
