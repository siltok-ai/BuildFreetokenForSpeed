"""监督单个验收进程树并采样Windows commit及NVML；不建立CUDA上下文。"""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import time

import psutil


class PerformanceInfo(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
        ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',
         'KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(name,wintypes.DWORD) for name in
        ('HandleCount','ProcessCount','ThreadCount')]


class NvMemory(ctypes.Structure):
    _fields_=[(name,ctypes.c_ulonglong) for name in ('total','free','used')]


def main():
    """所有清理仅限本脚本启动的进程树，资源不足时保存原因并停止该任务。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--cwd',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=1800)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    record=args.prefix.with_suffix('.resource.json')
    if record.exists(): raise FileExistsError(record)
    nvml=ctypes.CDLL('nvml.dll')
    if nvml.nvmlInit_v2()!=0: raise RuntimeError('NVML init failed')
    handle=ctypes.c_void_p()
    if nvml.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(handle))!=0: raise RuntimeError('NVML GPU lookup failed')
    get_perf=ctypes.WinDLL('psapi').GetPerformanceInfo
    report=dict(command=command,cwd=str(args.cwd),started=time.time(),samples=0,
        min_commit_available=None,min_ram_available=None,min_gpu_free=None,
        max_commit_used=0,max_gpu_used=0,max_tree_rss=0,max_tree_private=0,stop_reason=None)
    start=time.monotonic();low=0;tree=[]
    with args.prefix.with_suffix('.stdout.log').open('w',encoding='utf-8') as stdout, \
         args.prefix.with_suffix('.stderr.log').open('w',encoding='utf-8') as stderr, \
         args.prefix.with_suffix('.samples.jsonl').open('w',encoding='utf-8') as samples:
        proc=subprocess.Popen(command,cwd=args.cwd,stdout=stdout,stderr=stderr,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        report['pid']=proc.pid
        while proc.poll() is None:
            info=PerformanceInfo();info.cb=ctypes.sizeof(info)
            if not get_perf(ctypes.byref(info),info.cb): raise ctypes.WinError()
            memory=NvMemory()
            if nvml.nvmlDeviceGetMemoryInfo(handle,ctypes.byref(memory))!=0: raise RuntimeError('NVML memory sample failed')
            commit=info.CommitTotal*info.PageSize
            available=(info.CommitLimit-info.CommitTotal)*info.PageSize
            ram=info.PhysicalAvailable*info.PageSize
            rss=private=0
            try: tree=[psutil.Process(proc.pid),*psutil.Process(proc.pid).children(recursive=True)]
            except psutil.NoSuchProcess: tree=[]
            for member in tree:
                try:
                    mi=member.memory_info();rss+=mi.rss;private+=mi.private
                except psutil.NoSuchProcess: pass
            row=dict(t=time.monotonic()-start,commit_used=commit,commit_available=available,
                     ram_available=ram,gpu_used=memory.used,gpu_free=memory.free,tree_rss=rss,tree_private=private)
            samples.write(json.dumps(row)+'\n');samples.flush();report['samples']+=1
            for key,value in [('min_commit_available',available),('min_ram_available',ram),('min_gpu_free',memory.free)]:
                report[key]=value if report[key] is None else min(report[key],value)
            for key,value in [('max_commit_used',commit),('max_gpu_used',memory.used),('max_tree_rss',rss),('max_tree_private',private)]:
                report[key]=max(report[key],value)
            low=low+1 if available<2**30 or ram<2*2**30 or memory.free<256*2**20 else 0
            if low>=3 or time.monotonic()-start>args.timeout:
                report['stop_reason']='resource_floor' if low>=3 else 'timeout'
                for member in reversed(tree):
                    try: member.kill()
                    except psutil.NoSuchProcess: pass
                break
            time.sleep(.5)
        report['returncode']=proc.wait();report['elapsed_s']=time.monotonic()-start
    remaining=[member for member in tree if member.is_running()]
    report['remaining_before_cleanup']=[member.pid for member in remaining]
    for member in reversed(remaining):
        try: member.kill()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(remaining,timeout=5)
    report['remaining_descendants']=[member.pid for member in remaining if member.is_running()]
    record.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
    raise SystemExit(report['returncode'] if report['returncode'] else (1 if report['stop_reason'] else 0))


if __name__=='__main__': main()
