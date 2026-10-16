"""复用GPU动态副本的适应性所有者交换，保留source5发布协议与完整原始专家。"""
import time
import numpy as np
import torch
import triton
from freetoken.moe import q2_adaptive as base
from freetoken.moe.q2_adaptive import _publish_exchange
from q2_cached_exchange_kernel_v1 import exchange


class CachedQ2Owners(base.AdaptiveQ2Owners):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cached_copy_counts=torch.zeros(2,device=self.cache.device,dtype=torch.int64)

    def adapt(self):
        """Drain users, exchange bytes, then publish tables in place so graph addresses remain valid."""
        start=time.perf_counter()
        c=self.cache
        p=c.protected_payload
        torch.cuda.synchronize(c.device)
        counts=c.decode_freq.cpu().numpy()
        self.heat+=counts-self.last_counts
        self.last_counts=counts.copy()
        swaps=self.candidates(self.heat)
        self.heat*=self.decay
        if not swaps:
            self.rounds.append(dict(calls=self.calls,swaps=0,wall_ms=(time.perf_counter()-start)*1000))
            return
        initial=p.initial_map.numpy()
        staged=[]
        host_pointers=np.zeros((len(c.bank_schema),self.max_swaps),dtype=np.int64)
        gpu_slots=np.zeros(self.max_swaps,dtype=np.int32)
        commands=np.zeros((self.max_swaps,4),dtype=np.int32)
        for i,(gain,layer,cold,hot) in enumerate(swaps):
            slot=p.protected_slots[(layer,hot)]
            host_index=p.host_indices[layer][cold]
            for bank,role in enumerate(c.bank_schema):
                host_pointers[bank,i]=self.pointer_tables[layer][bank,cold]
            gpu_slots[i]=slot
            commands[i]=(layer,cold,hot,slot)
            staged.append((i,layer,cold,hot,slot,host_index))
        self.source_ptrs.copy_(torch.from_numpy(host_pointers))
        self.gpu_slots.copy_(torch.from_numpy(gpu_slots))
        self.commands.copy_(torch.from_numpy(commands))
        self.count.fill_(len(swaps))
        exchange[(len(c.bank_schema),32)](c._copy_dst_ptrs,self.source_ptrs,
            c._copy_feat_bytes,self.gpu_slots,self.count,self.commands,c.slot_for_id.view(-1),
            self.cached_copy_counts,self.max_swaps,32,1024,p.experts,p.protected_begin,num_warps=8)
        _publish_exchange[(1,)](self.commands,self.source_ptrs,c._copy_dst_ptrs,c._copy_feat_bytes,
            self.owner_table_ptrs,c.slot_for_id.view(-1),c.id_of_slot,c.usage,
            c._protected_initial_map.view(-1),c._protected_reverse_map,self.count,
            p.experts,p.protected_begin,self.max_swaps,len(c.bank_schema),
            triton.next_power_of_2(self.max_swaps),num_warps=4)
        for i,layer,cold,hot,slot,host_index in staged:
            for bank,role in enumerate(c.bank_schema):
                # WDDM needs the device-mapped host address, not the CPU virtual address.
                old_host=self.pointer_tables[layer][bank,cold]
                self.pointer_tables[layer][bank,cold]=c.bank_caches[role].data_ptr()+slot*c.bank_caches[role][0].numel()*c.bank_caches[role].element_size()
                self.pointer_tables[layer][bank,hot]=old_host
            initial[layer,hot]=-1
            initial[layer,cold]=slot
            del p.protected_slots[(layer,hot)]
            p.protected_slots[(layer,cold)]=slot
            p.host_indices[layer][cold]=-1
            p.host_indices[layer][hot]=host_index
        torch.cuda.synchronize(c.device)
        self.rounds.append(dict(calls=self.calls,swaps=len(swaps),wall_ms=(time.perf_counter()-start)*1000,
                               gain_sum=sum(s[0] for s in swaps)))


def enable(max_swaps=96):
    """显式实验容量；调用者传入的组件测试容量保留，不覆盖它。"""
    class SelectedCachedOwners(CachedQ2Owners):
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('max_swaps',max_swaps)
            super().__init__(*args,**kwargs)
    base.AdaptiveQ2Owners=SelectedCachedOwners
