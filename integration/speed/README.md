# 当前材料与历史消费合同

当前选择统一读取 `current-release.json`，不要按文件名排序或默认使用 `consumer-contract.json`。

- 当前profile：`strataq2mtp2`，版本 `0.1.3+speed.strataq2mtp2`，MTP默认关闭。
- 当前消费证据：`consumer-contract-strataq2mtp2.json`；其producer_engine_commit是不可变引擎材料来源，不要求随供应工具提交改变。
- `consumer-contract.json`、`consumer-contract-q2opt2.json`、`consumer-contract-strataq2dev7.json`等保留各自历史版本与验收范围。
- `presets/flash-next-sm120-24g.json`为显式选择的本机性能方案。资源安全余量不是通用产品默认值。
- 用户权重/配置不在本仓库：通过 `tools/flash_next_weights.py`、`tools/flash_next_mtp.py`、`tools/flash_next_speed_config.py`生成，见仓库README和当前交付流程。
