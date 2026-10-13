"""隔离运行实验并采样 Windows commit；只终止本探针启动的进程树。"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

class PerformanceInfo(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
        ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,ctypes.c_ulong) for n in ('HandleCount','ProcessCount','ThreadCount')]

def memory():
    """读取系统提交量，避免将物理可用内存错当 commit 余量。"""
    info=PerformanceInfo();info.cb=ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info),info.cb): raise ctypes.WinError()
    return dict(commit_used=info.CommitTotal*info.PageSize,commit_limit=info.CommitLimit*info.PageSize,
        physical_available=info.PhysicalAvailable*info.PageSize)

class Nvml:
    """轻量读取设备遥测，不创建第二个 CUDA 上下文。"""
    def __init__(self):
        self.lib=ctypes.CDLL('C:/Windows/System32/nvml.dll')
        if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML initialization failed')
        self.handle=ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))!=0:raise RuntimeError('NVML device lookup failed')
    def sample(self):
        result={}
        for label,method,extra in [('temperature','nvmlDeviceGetTemperature',[0]),('power_mw','nvmlDeviceGetPowerUsage',[]),
                ('sm_clock_mhz','nvmlDeviceGetClockInfo',[1]),('memory_clock_mhz','nvmlDeviceGetClockInfo',[2])]:
            value=ctypes.c_uint()
            code=getattr(self.lib,method)(self.handle,*extra,ctypes.byref(value))
            if code==0:result[label]=value.value
        return result

def main():
    """保留命令、所有采样和退出码；容量越界中止并记录失败，不改系统配置。"""
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--projected-gib',type=float,default=50)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command[0]=='--' else args.command
    baseline=memory();report=dict(command=command,baseline=baseline,samples=[],sampling_interval_seconds=.1)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if baseline['commit_limit']-baseline['commit_used']<(args.projected_gib+1)*1024**3:
        report.update(returncode=None,admission_rejected=True,reason='projected commit growth plus 1 GiB safety exceeds current limit')
        path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report));return 3
    start=time.perf_counter();nvml=Nvml();next_gpu=0
    with path.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,path.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,stdout=out,stderr=err)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            sample=memory(); sample['seconds']=time.perf_counter()-start; sample['processes']=[]
            if sample['seconds']>=next_gpu:
                sample['gpu']=nvml.sample();next_gpu=sample['seconds']+1
            try:
                owned_processes=[process,*process.children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            for p in owned_processes:
                try:
                    m=p.memory_info();cpu=p.cpu_times();io=p.io_counters()
                    sample['processes'].append(dict(pid=p.pid,rss=m.rss,private=getattr(m,'private',None),
                        cpu_user=cpu.user,cpu_system=cpu.system,read_bytes=io.read_bytes,write_bytes=io.write_bytes))
                except(psutil.NoSuchProcess,psutil.AccessDenied):pass
            report['samples'].append(sample)
            if sample['commit_limit']-sample['commit_used']<1*1024**3:
                report['aborted']='commit safety fell below 1 GiB'
                for p in reversed(process.children(recursive=True)):
                    try:p.terminate()
                    except psutil.NoSuchProcess:pass
                process.terminate();break
            time.sleep(.1)
        report.update(returncode=child.wait(),wall_seconds=time.perf_counter()-start)
    report['peak_commit_used']=max(s['commit_used'] for s in report['samples'])
    report['min_commit_available']=min(s['commit_limit']-s['commit_used'] for s in report['samples'])
    report['min_physical_available']=min(s['physical_available'] for s in report['samples'])
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False),flush=True)
    return 0 if report['returncode']==0 else 2

if __name__=='__main__':raise SystemExit(main())
