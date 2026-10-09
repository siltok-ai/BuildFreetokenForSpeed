# Speed 消费入口

这里保存机器可读的 [consumer-contract.json](consumer-contract.json) 与相对路径的 [manifest 示例](examples/)。完整解释见 [接入约定](../../docs/speed-integration.md)。

Speed 自身的源码适配不在这里复制。最终适配提交为 [808003f0](https://github.com/siltok-ai/siltok-speed/tree/808003f0efc70e7f43226e5d1e8c32b3704405cb)，随包验收与材料身份见 [最终记录](../../artifacts/win1-cp312-cu130-sm120/validation/packaged-speed-20261009/record.json)。原探索基线和快照 SHA 仍保存在历史 [validation-summary](../../artifacts/win1-cp312-cu130-sm120/validation-summary.json) 中。

已有 wheel 内含默认关闭的实验代码。普通模型与 INVALID 实验必须使用独立条目，不把实验参数放进全局 worker 环境。
