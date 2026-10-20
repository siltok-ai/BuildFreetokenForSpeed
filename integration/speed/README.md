# 当前材料与历史消费合同

[current-release.json](current-release.json) 是唯一选择入口，字段绑定profile、materials、consumer_contract、preset和输入SHA。`consumer-contract.json` 是历史win1合同，文件名顺序不表示当前版本。

Speed 开发入口为 `feat/flash-next`。当前 portable1 消费提交为 `79c39410`，主模型配套信息从 GGUF 导出，官方 MTP 预置在同一资源包中；当前资格及范围由上述指针选择的合同记录。历史 external 合同和安装包仍绑定原整合分支的 `02074c99`，不因新材料发布而改写；历史身份与 `--speed-root` 限制见 [源码分支与验收身份](../../docs/speed-source-history.md)。

- portable1：GGUF 独立资源包与相对路径加载，沿用相同3native/27AOT；使用流程和本轮验证见 [GGUF资源包交付](../../docs/flash-next-gguf-bundle.md)。
- external：沿用FT9原材料，Speed移除qwen36包/源码AOT DLL并消费明确外部cache；源码/普通wheel/保护包/实际i08安装分别封存，实际卸载及15项清理通过，见[历史交付](../../docs/flash-next-external-final-20261019.txt)。此前perf9、runtime2及整体diff review合同保留各自范围。
- review2b：晚间8项边界/生命周期修复，55组补丁；源码、普通wheel与隔离保护载荷分别验收，未实装，见[记录](../../docs/flash-next-review2-20261017.txt)。
- review1：perf8后追加正确性与维护性修订，范围见 [本轮review](../../docs/flash-next-review-20261017.txt)。
- perf8：普通wheel、Speed源码、保护载荷及真实安装分别封存；旧安装资格不会自动转移到review1。
- perf2/3/5/6、mtp2、dev7、q2opt2、win1等合同均保留其原材料和各自范围。
- weight source lock和MTP manifest为引擎输入身份，用户checkpoint与运行路径由准备工具生成，不把本机绝对路径写入发布锁。

当前使用步骤见 [GGUF 资源包流程](../../docs/flash-next-gguf-bundle.md)，旧 FT9 的准备流程仅用于历史复现。
