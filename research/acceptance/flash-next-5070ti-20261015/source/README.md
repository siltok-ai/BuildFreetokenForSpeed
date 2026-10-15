# 5070 Ti 源码阶段原始证据

2026-10-15，完整两片GGUF、冻结FreeToken strataq2mtp2及配套27 AOT；不修改引擎。
结果说明见 [交接实测](../../../../docs/flash-next-5070ti-acceptance.md)。

`index.json`逐文件固定字节与SHA，`python -B tools/verify_flash_next_5070ti.py`
从原始SSE到达时间重新计算18次512-token性能，核对六组MTP同容量输出、
四档20题历史token IDs/三题thinking、取消复用，以及正式CLI三种配置的运行退出。
监督采样、stdout/stderr、命令行、固定提示、实际runtime代码及AOT身份均保留。
`source-at-performance.patch`绑定各探针记录的历史HEAD9791fc7c与未提交diff；
`source-at-cli.patch`另含之后的文档修订。最终提交和安装包将在交付合同中另行绑定，
不能把本目录的源码阶段身份当成最终干净提交或安装包。

TCP benchmark/smoke的外部测试进程使用已授权的进程内授权替身；没有修改产品认证文件。
正式CLI三模式没有注入替身，运行真实serve模块、生成manifest、NVML准入和TCP请求，
通过隐藏控制台Ctrl+C正常退出。首次CLI v1因监督子进程使用裸python而命中了基础解释器，
在模型启动前缺少httpx；v2固定绝对venv解释器，未改变产品或安装依赖。

质量基线的五个历史失败保留，15/20不代表所有题都通过。新机不要求达到5090 D v2历史速度。
旧35B及H3源码结果保留在本机runs，H3大视频不复制进Git；后续安装包必须重新验证。
本目录不能证明Cython整包、安装器、整合分支、全新OS、离线依赖闭包或长时间压力验收。
