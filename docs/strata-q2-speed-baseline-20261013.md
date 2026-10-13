# 完整 Strata Q2 的 Speed 无 MTP 基线

FreeToken `0.1.3+speed.strataq2dev7` 的引擎/材料提交为 `ab06451cc100823ce926d98bef9db004c1e54234`。Speed运行源码提交 `23311dbd0611647671c41692a7113b738597d852`，证据提交 `a0fc86d9a8278c1c815893b319d25ae952b0ce4e`，各自位于独立研究分支，main未合并。精确wheel、27个AOT、完整GGUF两分片、资源配置和验收范围见[消费合同](../integration/speed/consumer-contract-strataq2dev7.json)。旧profile和合同保留。

真实TCP/SSE，在历史70-token和4096-token提示上，各三次实际512-token输出，中位109.88/104.02 token/s，最低109.58/103.85，满足本机单请求目标，无MTP。独立引擎十二次512-token运行中位112.63/103.42；直接backend使用另一组提示中位113.47/111.52。不同提示、EOS和缓存口径不能混在一起比较。Strata更高的无MTP结果仍是进一步改进的参考。

已通过安装后的Speed Python wheel真实模型验证、包内192个Python文件身份匹配、27个AOT运行核验、实际host/GPU专家所有权、HTTP四项功能质量、取消后worker复用和退出清场，以及314项Speed相关测试。历史混合精度Flash Next的20题输出token全部与历史记录相同，16/20。4096槽旧模型回归触及commit监控，保留中止记录；1024槽完成质量回归，不能据此宣称4096槽资源验收。完整Windows安装器、干净机器、其他设备及两种历史模型本轮实际回归仍未验收。

当前使用永久驻留9216、动态1024、prefill1024。Speed主进程用显式NVML探测避免额外CUDA上下文。经实测的实验准入参数与默认安全策略严格区分，不把有限显存余量推广到其他模型或机器。原始运行资料及失败记录位于Speed的 `research/strata-q2-dev7-20261013`，gzip前后SHA均在索引中。

下一项结构优化是MTP，独立分支保留无MTP基线。GGUF没有预测层，需要固定revision的31个BF16来源张量（实查5,214,301,696字节，28个分片）。要适配预测层、目标模型的连续小窗口验证、GDN/conv/KV/PLE状态事务、取消/前缀缓存、采样兼容和资源重预算。减少专家驻留会抵消部分草稿收益，必须同时测原11264槽无MTP、缩小驻留无MTP、相同预算MTP三组最终交付速度。Prefill TTFT回退继续跟踪。
