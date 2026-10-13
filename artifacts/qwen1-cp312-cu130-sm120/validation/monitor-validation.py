"""采样本次子进程树和整机容量；不修改系统配置或其他进程。"""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import time
import psutil

class PerformanceInfo(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in (
        'CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',
        'KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,wintypes.DWORD) for n in ('HandleCount','ProcessCount','ThreadCount')]
class GpuMemory(ctypes.Structure):
    _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
psapi=ctypes.WinDLL('psapi')
nvml=ctypes.WinDLL('nvml.dll')
if nvml.nvmlInit_v2()!=0: raise RuntimeError('NVML initialization failed')
gpu=ctypes.c_void_p()
if nvml.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(gpu))!=0: raise RuntimeError('GPU lookup failed')
def sample():
    """使用 Windows 与 NVIDIA 原生接口读当前容量。"""
    info=PerformanceInfo(); info.cb=ctypes.sizeof(info)
    if not psapi.GetPerformanceInfo(ctypes.byref(info),info.cb): raise ctypes.WinError()
    memory=GpuMemory()
    if nvml.nvmlDeviceGetMemoryInfo(gpu,ctypes.byref(memory))!=0: raise RuntimeError('GPU memory query failed')
    return {'commit_used':info.CommitTotal*info.PageSize,'commit_available':(info.CommitLimit-info.CommitTotal)*info.PageSize,
            'physical_available':info.PhysicalAvailable*info.PageSize,'gpu_used':memory.used,'gpu_free':memory.free}
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
baseline=sample(); observations=[baseline];seen=set();started=time.perf_counter()
child=subprocess.Popen(command);root=psutil.Process(child.pid)
while child.poll() is None:
    s=sample(); rss=private=0
    try: processes=[root,*root.children(recursive=True)]
    except psutil.NoSuchProcess: processes=[]
    for proc in processes:
        try:
            m=proc.memory_info(); rss+=m.rss;private+=m.private;seen.add(proc.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    s.update(tree_rss=rss,tree_private=private);observations.append(s);time.sleep(.1)
observations.append(sample())
result={'command':command,'returncode':child.returncode,'interval_seconds':.1,'samples':len(observations),
        'baseline':baseline,'wall_seconds':time.perf_counter()-started,'owned_pids':sorted(seen),
        'peak_gpu_used':max(x['gpu_used'] for x in observations),'min_gpu_free':min(x['gpu_free'] for x in observations),
        'peak_commit_used':max(x['commit_used'] for x in observations),'min_commit_available':min(x['commit_available'] for x in observations),
        'min_physical_available':min(x['physical_available'] for x in observations),
        'peak_tree_rss':max(x.get('tree_rss',0) for x in observations),'peak_tree_private':max(x.get('tree_private',0) for x in observations)}
a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'capacity_monitor':result}),flush=True)
nvml.nvmlShutdown()
raise SystemExit(child.returncode)
