# 维护当前 FreeToken 交付

## 职责

本仓库维护 FreeToken 定制、源码输入、wheel/AOT 构建和 Flash Next 资源准备。Speed 维护模型准入、worker、HTTP、授权、服务和安装器。完整 Speed 打包环境由打包者自行维护。

`upstream/FreeToken` 是未经修改的官方快照。使用当前 profile 重放到独立目录，在重放结果中开发；不要临时修改共享 site-packages。

## 更新源码与材料

1. 从 [current-release.json](../integration/speed/current-release.json) 读取现用 profile，运行 `tools/prepare_source.py`。
2. 将引擎改动保存为新增补丁，建立新 catalog、profile 和修订版本，保持既有 patch/wheel 字节。
3. 仅 Python 变化且符合复用边界时可重新封装；原生、AOT、公共头文件或 ABI 变化时完整构建。
4. 运行仓库检查和实际失败场景的回归，再验证 GPU 组件、Speed OFF/ON、HTTP/SSE 与卸载回收。
5. 更新当前指针、配套 Speed release 和必要文档；构建新安装器时按影响范围完成安装测试。

文档和工具整理不改变引擎版本。改变工具链时使用独立 profile 和输出目录，验证新产物，不覆盖当前材料。

## 检查范围

```powershell
python -m pip install -r tools/requirements-test.txt
python -B tools/validate_repository.py
python -B -m unittest discover -s tests -v
```

日常检查只针对当前交付及其必要二进制复用基线。实机结果保存在仓库外；保留真实失败及中断记录，不把旧版本的实机结果当作新版本已通过。

修改资源制作工具时，从固定来源在独立目录重新制作，检查 GGUF、MTP、元数据、manifest 和资源配置的一致性，并用匹配 Speed 验证 OFF/ON。比较性能时固定设备、请求、实际输出长度、专家/KV 预算及 MTP 策略。

## 更新上游与回滚

使用 `tools/export_upstream.py --repo <checkout> --commit <完整commit> --output <新目录>` 导出候选。逐项判断补丁是否仍需要，处理冲突后建立新 profile；不得直接修改浏览快照或自动向官方仓库提交。

回滚时恢复配套的 Speed release、FreeToken 材料和模型配置。旧版本从 Git 历史获取；开发探索、性能轨迹和安装日志放在仓库外，README 维护当前步骤和必要限制。

消费检查只校验当前源码、材料和兼容性，实机验收独立进行。文档维护当前可用步骤，不要求消费方访问维护机器的目录或使用特定 Speed 分支。
