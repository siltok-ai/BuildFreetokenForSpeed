"""Read-only CUDA alias for owned mapped host bytes; the CPU owner outlives every graph."""
import torch
from freetoken.kernel.pinned import device_ptr


class MappedHostBytes:
    def __init__(self,owner):
        """Expose the device-visible address, including Windows' distinct mapped VA."""
        if owner.device.type!='cpu' or owner.dtype!=torch.uint8 or not owner.is_contiguous():
            raise ValueError('Mapped bytes require contiguous pinned CPU uint8 storage')
        self.owner=owner
        self.ptr=device_ptr(owner)
        self.__cuda_array_interface__=dict(shape=(owner.numel(),),strides=None,typestr='|u1',
                                          data=(self.ptr,False),version=3)

    def alias(self,device):
        """Require zero-copy import; a copy would silently defeat the memory experiment."""
        result=torch.as_tensor(self,device=device)
        if result.data_ptr()!=self.ptr or not result.is_cuda or result.dtype!=torch.uint8:
            raise RuntimeError('CUDA array import did not preserve the mapped host address')
        return result

