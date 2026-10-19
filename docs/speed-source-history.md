# Speed源码分支与验收身份

2026-10-19，Speed原整合分支相对dev的20笔提交压缩为新分支的1笔提交。原分支及其历史保持不变；本次仅整理提交历史，没有修改产品文件。

| 用途 | 分支 | 固定提交 |
| --- | --- | --- |
| 后续源码开发入口 | `feat/flash-next` | `abc03e1114d2d7f6f7dfb05d617f237f34782f9f` |
| 保留原20笔历史；当前合同与安装包的来源 | `integrate/flash-next-dev-20261018` | `02074c99d8c4b922cb063cdb4bb84dfef8172ae6` |
| 压缩时的基线；新提交的唯一父提交 | `dev` | `c89720fe4d5c4793eed7d83e3f56ccbda38e1f08` |

表中提交为本次整理时的固定身份，后续分支推进时应重新核对。两个交付源码提交的完整Git tree均为 `c864f2247f1e833611fcee77e60bb2395915a573`，包含文件内容、文件模式和全部5个子模块指针。相对固定dev基线的完整binary diff也逐字节相同，SHA-256为 `d9a178b557d64e869171be490015d796bf167b30a59fb2668888a0622528c319`。

## 材料与验收范围

[当前选择](../integration/speed/current-release.json)保持 `0.1.3+speed.strataq2perf9`，57组补丁、wheel内3个native模块及配套27项AOT不变。Speed的generic release仍锁定原供应提交和实际wheel SHA；打包者继续提供本仓库的匹配wheel和外部kernel-cache。新源码分支不包含旧 `silitok_speed/qwen36` 目录或AOT DLL，历史公开模型ID及manifest backend别名仍兼容。

现有 [实际安装合同](../integration/speed/consumer-contract-strataq2perf9-external-installed.json)、其源码/编译合同和 [实装报告](flash-next-external-final-20261019.txt)仍记录 `02074c99`。源码树一致与实际安装验收是两个独立事实；没有把原安装包的构建SHA改成 `abc03e11`，也没有因历史压缩重跑GPU或安装测试。已记录的MTP观测下降及长输出问题继续保留。

`python -B tools/verify_current_release.py` 不接收Speed checkout参数，只复核当前材料、合同和冻结证据。在消费验证器中传入 `--speed-root` 会额外检查checkout的精确HEAD；旧external合同要求 `02074c99`，因此传入 `abc03e11` 的checkout会以 `Speed checkpoint commit changed` 拒绝。这是预期的来源检查，不能通过修改旧合同或放宽检查规避。复核旧合同使用原提交的checkout；从新提交构建交付时另记真实身份和验收范围。

## 复核源码等价性

在自己的Speed checkout执行；示例路径可按本机替换，无需本机开发目录或安装路径：

```powershell
$speed = 'D:\src\siltok-speed'
git -C $speed fetch origin dev feat/flash-next integrate/flash-next-dev-20261018
$original = '02074c99d8c4b922cb063cdb4bb84dfef8172ae6'
$squashed = 'abc03e1114d2d7f6f7dfb05d617f237f34782f9f'
$base = 'c89720fe4d5c4793eed7d83e3f56ccbda38e1f08'

# 两行均应输出上文的同一tree；比较的是固定提交，不受本地pyc等忽略文件影响。
git -C $speed rev-parse "${original}^{tree}" "${squashed}^{tree}"
# 无输出且退出码为0；文件内容、模式及子模块指针无差异。
git -C $speed diff --exit-code $original $squashed
# 依次应为20、1，最后应输出固定的$base（唯一父提交）。
git -C $speed rev-list --count "${base}..${original}"
git -C $speed rev-list --count "${base}..${squashed}"
git -C $speed show -s --format=%P $squashed
```

tree一致不包括未跟踪的本地文件、Python环境、模型或材料路径。新checkout仍需初始化构建所需子模块并准备配套依赖和外部材料；它们的操作步骤见 [当前交付](flash-next-current-delivery.md)与Speed的Windows打包说明。
