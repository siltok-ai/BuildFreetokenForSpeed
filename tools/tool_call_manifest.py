"""只为显式选择的新部署声明 FreeToken 函数调用能力。"""


def enable_function_tools(entry):
    """保持权重和容量配置，声明需要支持 qwen3_coder 的新版 Speed。"""
    entry["features"].append("tools")
    entry["tool_call_format"] = "qwen3_coder"
