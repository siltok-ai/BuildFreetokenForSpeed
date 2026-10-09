# 最终 Speed 随包验收

本目录追加封存 2026-10-09 的实际打包与三模型运行证据，原源码验收和材料构建记录保持不变。

Speed 提交：[808003f0efc70e7f43226e5d1e8c32b3704405cb](https://github.com/siltok-ai/siltok-speed/commit/808003f0efc70e7f43226e5d1e8c32b3704405cb)，分支 `feat/freetoken-win1-flash-next`。使用本仓库 commit `12bdf743bbf4725fc25468204313cb2f43733758` 中已有的 `0.1.3+speed.win1` wheel 与 26 DLL，没有重建 FreeToken 或增加引擎补丁。

## 阅读入口

| 文件 | 内容 |
| --- | --- |
| [record.json](record.json) | 最终 Speed commit/tree、材料 commit/wheel SHA、验证范围与未验收边界 |
| [acceptance-summary.json](acceptance-summary.json) | 三模型 HTTP、短长请求、取消复用和诊断性能摘要 |
| [packaged-models-summary.json](packaged-models-summary.json) | 六组实际命令、退出码与执行时长 |
| [compiled-source-proof.json](compiled-source-proof.json) | 编译后的 208 个模块与待提交源码 SHA 全匹配 |
| [packaging-inventory-summary.json](packaging-inventory-summary.json) | Speed 单 DLL SHA、26 cache DLL、零 `.pyc`、manifest/lock 位置 |
| `packaged-<profile>-<http/backend>.json` | 每组原始运行报告 |
| `packaged-<profile>-<http/backend>-identity.json` | 每组随包 Python/Speed/FreeToken 路径及完整编译 manifest |
| `packaged-*.log`、[packaging.log](packaging.log)、[targeted-tests.log](targeted-tests.log) | 实际模型、完整打包和 199 项定向测试日志 |
| [packaged_child.py](packaged_child.py)、[run_packaged_models.py](run_packaged_models.py) | 当时执行的仓库外验收脚本，保留原始本机路径 |
| [evidence.json](evidence.json) | 原始报告、日志、脚本和记录的逐文件 SHA/大小 |

脚本是本次运行经过的封存证据，使用原本机 app、Speed checkout 和权重位置；它们不是本仓库的通用构建入口。重新构建素材继续使用 `build/build-materials.ps1`。原始 Speed 应用源码和 patch 不在本仓库复制；record 中的 reviewed patch SHA 用于关联 Speed 的外部 review 记录。

## 结论与边界

完整 `build.ps1 -SkipInstaller` 成功，三模型各完成 HTTP/SSE 和 backend 两组运行，短长请求、取消后同 worker 复用及退出通过。随包 native 三个 `.pyd`、440 个 FreeToken Python 文件与既有 wheel 逐字节一致，26 个 DLL 与既有材料 SHA 一致。这些补充核对说明本轮消费的是原材料，没有从外部环境借代码。

Flash Next 使用 20 物理专家层服务 48 逻辑层、SSD PLE，生成语义无效。两轮 32 输出 token 的诊断中位 TPS：Qwen3.6 164.32、Qwen3.8-27B 57.85、Flash Next INVALID 38.19；不是稳定 A/B。此前源码运行的 RSS 峰值不作为这次打包的重新实测。

打包发生在最终提交前，因此 manifest 记录基线 commit 和 `dirty=true`，逐模块 SHA 绑定最终代码，未伪装成提交后重新构建。清理 PATH 是同一本机的运行环境隔离；没有生成安装器、进行干净机器安装或验收完整正确 Flash Next。

`python -B tools/validate_repository.py` 会校验消费合同、记录与证据 SHA 的关联；它不会在无 GPU 的 CI 中重新执行模型或声称安装验收通过。
