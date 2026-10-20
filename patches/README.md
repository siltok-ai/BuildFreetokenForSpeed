# FreeToken 定制补丁

当前有序目录由 [current-release.json](../integration/speed/current-release.json) 指向的 profile 选择，即 [strataq2portable1-series.json](strataq2portable1-series.json)。它包含 58 个补丁；构建按 JSON 顺序应用，不能按文件名排序，也不能只应用最后一个补丁。

## 当前功能

| 补丁范围 | 作用 |
| --- | --- |
| 1–6 | Windows runtime、原生/AOT 构建、Q2 专家执行及包版本 |
| 7–13 | 完整两片 GGUF 加载、主干/PLE、批量算术、原生解码及专家驻留 |
| 14–20 | MTP 草稿、前缀状态、事务所有权、greedy 调度与取消 |
| 21–48 | Q2 预填充、紧凑状态、专家缓存、主机存储及解码执行 |
| 49–57 | stop/采样/输出预算边界，以及主存加载和 Windows Q2 等待策略 |
| 58 | checkpoint 相对 GGUF 路径与可搬迁加载 |

每项的 `title`、`files` 和 patch 正文给出具体改动。补丁文件名保留原标识，部分名称沿用开发阶段命名。当前功能由完整序列叠加形成。

## 重放和修改

```powershell
python -B tools/prepare_source.py --output D:\builds\ft-source
```

在输出目录的 `build-source` 中开发，不直接修改 `upstream/FreeToken`。需要提交引擎改动时：

1. 以当前重放结果为基线生成新增补丁，保留既有补丁内容与顺序。
2. 新建有序 catalog、profile 和 FreeToken 修订版本，列明改变的文件。
3. 根据是否改变原生/AOT 输入选择重新封装或完整构建。
4. 完成工具检查及受影响的 GPU/Speed 验证，再更新当前指针。

[维护规则](../docs/maintenance.md) 说明输入、版本和检查要求。[构建入口](../build/README.md) 负责实际生成 wheel/AOT。`strataq2perf9-series.json` 仅用于当前二进制复用基线重放。
