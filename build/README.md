# 固定 Windows 构建

默认 profile 为 profiles/qwen1.json，规格为 specs/qwen-sm120.txt。使用 requirements.lock.txt 准备正常安装的独立 CPython 3.12 venv，不能用 .pth 借另一个环境。

运行 build-materials.ps1 的 -PreflightOnly 核对全部冻结输入、MSVC、CUDA、Torch 与依赖，再用新 OutDir 构建。输出带两个 .pyd 的 wheel、19 个 AOT DLL、不可变哈希清单和 build-record.json。

生产者不下载上游 main，不混入旧 DLL，失败输出留作诊断。编译成功的状态始终是 built_requires_runtime_validation；后续模型和随包验收追加为独立记录。
