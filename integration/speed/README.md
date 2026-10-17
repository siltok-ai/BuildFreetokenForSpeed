# 当前材料与历史消费合同

[current-release.json](current-release.json) 是唯一选择入口，字段绑定profile、materials、consumer_contract、preset和输入SHA。`consumer-contract.json` 是历史win1合同，文件名顺序不表示当前版本。

- review1：perf8后追加正确性与维护性修订，范围见 [本轮review](../../docs/flash-next-review-20261017.txt)。
- perf8：普通wheel、Speed源码、保护载荷及真实安装分别封存；旧安装资格不会自动转移到review1。
- perf2/3/5/6、mtp2、dev7、q2opt2、win1等合同均保留其原材料和各自范围。
- weight source lock和MTP manifest为引擎输入身份，用户checkpoint与运行路径由准备工具生成，不把本机绝对路径写入发布锁。

当前使用步骤见 [交付流程](../../docs/flash-next-current-delivery.md)。
