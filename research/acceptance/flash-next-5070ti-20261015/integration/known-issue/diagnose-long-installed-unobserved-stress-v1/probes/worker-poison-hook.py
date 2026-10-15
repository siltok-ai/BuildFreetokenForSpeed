"""诊断专用：把未初始化浮点分配填成NaN，暴露缺写或无效位置参与运算。"""
from pathlib import Path
import faulthandler
import os
import runpy

runpy.run_path(str(Path(__file__).with_name('worker-numeric-hook.py')), init_globals={'OUTPUT': OUTPUT})

import torch

trace_file = (Path(OUTPUT) / ('trace-' + str(os.getpid()) + '.txt')).open('w', encoding='utf-8')
faulthandler.dump_traceback_later(90, repeat=True, file=trace_file)

original_empty = torch.empty
original_empty_like = torch.empty_like
original_new_empty = torch.Tensor.new_empty
original_empty_strided = torch.empty_strided


def poison(tensor):
    """只处理本应由调用者完整写入的浮点empty，不覆盖zeros或任何已有输入。"""
    if tensor.is_floating_point():
        with torch.no_grad():
            tensor.fill_(float('nan'))
    return tensor


def empty(*args, **kwargs):
    return poison(original_empty(*args, **kwargs))


def empty_like(*args, **kwargs):
    return poison(original_empty_like(*args, **kwargs))


def new_empty(self, *args, **kwargs):
    return poison(original_new_empty(self, *args, **kwargs))


def empty_strided(*args, **kwargs):
    return poison(original_empty_strided(*args, **kwargs))


torch.empty = empty
torch.empty_like = empty_like
torch.Tensor.new_empty = new_empty
torch.empty_strided = empty_strided
