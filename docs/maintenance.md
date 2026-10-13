# 双模型维护合同

上游 commit → 最小平台补丁 → 新定制版本 → 完整 wheel/19 DLL → 两模型源码回归 → Speed dev 随包回归。

冻结来源为 0781324，保持官方归档与浏览树一致。每个补丁须列明文件和原因；上游已覆盖的功能不再重复实现。FP8 rowwise 替代路径、quant_backend 和请求三元组均来自官方。

本候选固定 Windows MoE 串行读取，不承诺 POSIX 并行路径可用。若以后增加并行读取，单独修订平台补丁与版本并记录性能。

生产者记录 wheel/cache 来源和实际字节身份；Speed 文档指向本分支，由构建者保证 wheel/cache 匹配，不增加材料 SHA 校验或运行时材料锁。证据反向记录 Speed commit，不用互相改 commit 制造循环。

不能把同版历史 DLL 拼入新输出，不能覆盖已验收 wheel，也不能将 import/库存检查写成推理通过。官方 FP8 与其他硬件不继承本机 NVFP4 验收。
