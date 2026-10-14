# 全GGUF与MTP供应流程验收 · 2026-10-14

功能提交 `9b3a793d033cecba5b937302f2a8f8be49695501`。`index.json`封存36份原始记录的原始SHA和gzip SHA；`tools/verify_flash_next_delivery.py`交叉核对转换与实际交付结果。本次引擎代码、FT2 wheel（20f4ea89...）及27个AOT不变。

- 从固定HF revision实际下载五份元数据并完整验真；不下载HF主干权重。
- 真实完整两片GGUF在新来源树重新SHA校验。大源文件通过只读HardLink复用磁盘数据，没有修改旧golden文件；不是另一台物理机测试。
- 新目录31份BF16 MTP来源重新验真、转换全部823,887,872字节。转换约277.98秒，来源verification SHA与旧目录不同；全部派生张量名称、形状、类型、大小、源/派生SHA与批准数据一致。安装固定runtime.json后SHA仍为610c51...，不将本机路径/时间引入新的发布身份。
- 模型树搬到另一目录，旧绝对路径正确被拒绝；refresh全量SHA校验后重建本机回执，实际Speed使用搬迁目录加载。配置与tokenizer遵循原验收元数据。
- 仅git staged tree导出，无未跟踪文件。导出副本29项测试全部通过，当前指针/材料通过，20补丁重放、wheel全456份Python及原生/AOT身份通过。完整仓库validator也通过。
- 安装环境使用`python -I`、FT2与Speed0.1.2.dev0+strataq2mtp2.1，显式当前AOT cache和全新Triton cache。生成的manifest走实际Speed schema、磁盘契约、discovery注册，SILITOK_CONFIG实际读取NVML/64MiB GPU/2GiB commit预算。
- MTP开启（7168/9216槽）和关闭（9216/11264槽）均答对527、正常EOS。各两次512-token交付逐轮匹配原HTTP对应轮次文本，四次token数量/usage/流事件一致，所有测试worker卸载退出。仅作新工具交付回归：on 112.13/103.96tps，off112.08/112.25tps；未重做完整HTTP性能统计。
- v2完整进程监督rc0，peak commit63,299,461,120字节，最小剩余commit7,058,399,232字节。没有改全局内存/缓存/服务配置。

v1探针把第二轮误配为第一轮参考，因此断言失败；两轮实际都匹配各自历史参考。保留v1脚本、结果与失败日志；v2修正逐轮比较后开/关全部通过。未通过修改引擎、放松比较或丢弃失败记录让验证通过。

权重源、派生大文件、Triton cache不进Git，使用仓库工具重建。已验收wheel/AOT仍在artifacts中。证据里的绝对路径仅为当次研究记录，不是新工具运行默认值。原始probe使用本机监督器和冻结prompt；对应脚本/输入一并归档。

此验收不覆盖新Speed Cython单DLL整包、安装器/干净机、完整离线依赖wheelhouse、其他GPU或操作系统。Speed导出后的行尾/包清单问题仍需消费端另行修复；本仓库main仅作为当前FreeToken材料与权重准备入口，不把这些未完成项目写成通过。
