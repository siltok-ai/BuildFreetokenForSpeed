"""Read this process's NVIDIA WDDM local/nonlocal budgets without changing reservations."""
import ctypes as C
from ctypes import wintypes as W
import json
import uuid


class GUID(C.Structure):
    _fields_=[('Data1',W.DWORD),('Data2',W.WORD),('Data3',W.WORD),('Data4',C.c_ubyte*8)]


class LUID(C.Structure):
    _fields_=[('LowPart',W.DWORD),('HighPart',W.LONG)]


class Desc1(C.Structure):
    _fields_=[('Description',W.WCHAR*128),('VendorId',W.UINT),('DeviceId',W.UINT),
        ('SubSysId',W.UINT),('Revision',W.UINT),('DedicatedVideoMemory',C.c_size_t),
        ('DedicatedSystemMemory',C.c_size_t),('SharedSystemMemory',C.c_size_t),
        ('AdapterLuid',LUID),('Flags',W.UINT)]


class MemoryInfo(C.Structure):
    _fields_=[(name,C.c_uint64) for name in ('Budget','CurrentUsage','AvailableForReservation','CurrentReservation')]


def guid(text):return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


def method(pointer,index,result,*arguments):
    table=C.cast(pointer,C.POINTER(C.POINTER(C.c_void_p))).contents
    return C.WINFUNCTYPE(result,C.c_void_p,*arguments)(table[index])


def checked(result):
    if result<0:raise OSError(f'DXGI HRESULT 0x{result&0xffffffff:08x}')


class BudgetReader:
    def __init__(self):
        self.adapter=C.c_void_p()
        factory=C.c_void_p()
        dll=C.WinDLL('dxgi')
        create=dll.CreateDXGIFactory1
        create.argtypes=[C.POINTER(GUID),C.POINTER(C.c_void_p)]
        create.restype=W.LONG
        checked(create(C.byref(guid('770aae78-f26f-4dba-a829-253c83d1b387')),C.byref(factory)))
        try:
            index=0
            while True:
                adapter=C.c_void_p()
                hr=method(factory,12,W.LONG,W.UINT,C.POINTER(C.c_void_p))(factory,index,C.byref(adapter))
                if hr<0:break
                index+=1
                try:
                    desc=Desc1()
                    checked(method(adapter,10,W.LONG,C.POINTER(Desc1))(adapter,C.byref(desc)))
                    if desc.VendorId!=0x10de:continue
                    checked(method(adapter,0,W.LONG,C.POINTER(GUID),C.POINTER(C.c_void_p))(
                        adapter,C.byref(guid('645967a4-1392-4310-a798-8053ce3e93fd')),C.byref(self.adapter)))
                    self.description=desc.Description
                    break
                finally:method(adapter,2,W.ULONG)(adapter)
        finally:method(factory,2,W.ULONG)(factory)
        if not self.adapter.value:raise RuntimeError('No NVIDIA DXGIAdapter3')

    def sample(self):
        result=dict(adapter=self.description)
        for segment,name in ((0,'local'),(1,'nonlocal')):
            info=MemoryInfo()
            checked(method(self.adapter,14,W.LONG,W.UINT,W.UINT,C.POINTER(MemoryInfo))(
                self.adapter,0,segment,C.byref(info)))
            result[name]={field:int(getattr(info,field)) for field,_ in info._fields_}
        return result

    def close(self):
        if self.adapter.value:
            method(self.adapter,2,W.ULONG)(self.adapter)
            self.adapter=C.c_void_p()


if __name__=='__main__':
    reader=BudgetReader()
    try:print(json.dumps(reader.sample()))
    finally:reader.close()
