# Speed源码分支与验收身份

2026-10-19，Speed整合分支压为dev之上的1笔提交，并将相关使用说明收敛到 `doc/speed/models/freetoken.md`。根 `docs/`、research阶段记录及旧MTP2专用核验脚本移出当前源码树，`.gitattributes`恢复dev策略。原20笔分支保持不变。

| 用途 | 分支 | 固定提交 |
| --- | --- | --- |
| 后续源码开发入口 | `feat/flash-next` | `a3e8b635870ca7e3dd02d8434fc87ca92582fc05` |
| 保留原20笔历史；当前合同与安装包的来源 | `integrate/flash-next-dev-20261018` | `02074c99d8c4b922cb063cdb4bb84dfef8172ae6` |
| 压缩时的基线；新提交的唯一父提交 | `dev` | `c89720fe4d5c4793eed7d83e3f56ccbda38e1f08` |

表中提交是本次维护的固定身份，后续分支推进时应重新核对。清理后的完整Git tree为 `3cd99c885eb4041c0352b7365efd6182287a7792`，与原分支不同；除 `backends/CONTEXT.md` 文档外，`silitok_speed/` 产品代码和资源保持原字节，全部5个子模块指针不变。

## 材料与验收范围

[当前选择](../integration/speed/current-release.json)保持 `0.1.3+speed.strataq2perf9`，57组补丁、wheel内3个native模块及配套27项AOT不变。Speed的generic release仍锁定原供应提交和实际wheel SHA；打包者继续提供本仓库的匹配wheel和外部kernel-cache。新源码分支不包含旧 `silitok_speed/qwen36` 目录或AOT DLL，历史公开模型ID及manifest backend别名仍兼容。

现有 [实际安装合同](../integration/speed/consumer-contract-strataq2perf9-external-installed.json)、其源码/编译合同和 [实装报告](flash-next-external-final-20261019.txt)仍记录 `02074c99`。本次通过mdBook、文档manifest注册、打包资源收集和真实外部材料自检，并另在全LF源码副本上复核；没有重跑整模型或安装测试，也没有改写旧安装包的构建SHA。已记录的MTP观测下降及长输出问题继续保留。

`python -B tools/verify_current_release.py`只复核当前材料、合同和冻结证据。在消费验证器中传入 `--speed-root` 会额外检查checkout的精确HEAD；旧external合同要求 `02074c99`，传入新提交会以 `Speed checkpoint commit changed` 拒绝。复核旧合同使用原提交的checkout；从新提交构建交付时另记真实身份和验收范围，不修改旧合同或放宽检查。

## 复核产品代码与单提交状态

在自己的Speed checkout执行；示例路径可按本机替换，无需本机开发目录或安装路径：

```powershell
$speed = 'D:\src\siltok-speed'
git -C $speed fetch origin dev feat/flash-next integrate/flash-next-dev-20261018
$original = '02074c99d8c4b922cb063cdb4bb84dfef8172ae6'
$squashed = 'a3e8b635870ca7e3dd02d8434fc87ca92582fc05'
$base = 'c89720fe4d5c4793eed7d83e3f56ccbda38e1f08'

# 均应无输出、退出码0；产品范围仅排除一份发生变更的说明文档。
git -C $speed diff --exit-code $original $squashed -- silitok_speed ':!silitok_speed/backends/CONTEXT.md'
git -C $speed diff --exit-code $base $squashed -- .gitattributes
# 依次应为20、1，最后应输出固定的$base（唯一父提交）。
git -C $speed rev-list --count "${base}..${original}"
git -C $speed rev-list --count "${base}..${squashed}"
git -C $speed show -s --format=%P $squashed
```

这些Git比较不包括未跟踪的文件或本机环境。新checkout仍需初始化构建所需子模块并准备配套依赖和外部材料，步骤见 [当前交付](flash-next-current-delivery.md)与Speed的Windows打包说明。
