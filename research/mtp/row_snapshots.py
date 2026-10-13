"""Diagnostic per-row state commit using leased FreeToken GDN slots; no installed-file edits."""
from __future__ import annotations

import types

import torch
from freetoken.core import get_global_ctx


class RowSnapshots:
    def __init__(self,engine,req,*,max_rows=4):
        """临时借用已分配但无所有者的slot，保存每行递推/卷积/PLE/ring，拒绝时直接提交。"""
        self.engine=engine
        self.req=req
        self.pool=engine.linear_state_pool
        self.slots=self.pool.alloc(max_rows)
        self.dst=torch.tensor(self.slots,device=engine.device,dtype=torch.int64)
        self.rings=torch.empty((max_rows,*engine.kv_cache._pending_ring.shape[1:]),
            device=engine.device,dtype=engine.kv_cache._pending_ring.dtype)
        self.live=req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        assert self.live not in self.slots
        import freetoken.models.qwen4_exp.gdn as gdn
        import freetoken.models.qwen4_exp.ple as ple
        self.gdn_class=gdn.Qwen4ExpGatedDeltaNet
        self.ple_class=ple.PLELayer
        self.ple_module=ple
        self.original_gdn=self.gdn_class._gguf_core
        self.original_ple=self.ple_class._write_track_snapshot
        self.original_ngram=ple.commit_ngram_context
        backend=engine.attn_backend
        self.original_index=backend._update_index_cache
        owner=self

        def gdn_core(op,conv,a,b,pool,li,fla,is_decode):
            if not getattr(fla,'_mtp_row_snapshots',False):
                return owner.original_gdn(op,conv,a,b,pool,li,fla,is_decode)
            old=pool.conv_states[li].index_select(0,fla.cache_indices.long())[0].transpose(0,1)
            history=torch.cat([old,conv],dim=0)
            output=owner.original_gdn(op,conv,a,b,pool,li,fla,is_decode)
            # 原chunk track只在完整历史之后取行；小窗口早期行必须包含进入窗口前的conv历史。
            n=old.shape[0]
            indices=fla.track_boundary_row[:,None]+torch.arange(n,device=conv.device)[None,:]
            states=history[indices].transpose(-1,-2).contiguous()
            pool.conv_states[li].index_copy_(0,fla.track_dst,states)
            return output

        def ple_snapshot(op,states,x,fla):
            if not getattr(fla,'_mtp_row_snapshots',False):
                return owner.original_ple(op,states,x,fla)
            old=states.index_select(0,fla.cache_indices.long())[0].transpose(0,1)
            history=torch.cat([old,x],dim=0)
            indices=fla.track_boundary_row[:,None]+torch.arange(op.state_len,device=x.device)[None,:]
            states.index_copy_(0,fla.track_dst,history[indices].transpose(-1,-2).contiguous())

        def ngram(meta,fla,context_pool=None):
            if not getattr(fla,'_mtp_row_snapshots',False):
                return owner.original_ngram(meta,fla,context_pool)
            if context_pool is None: context_pool=ple._ngram_context_pool()
            old=context_pool.index_select(0,meta.state_slots)[0]
            history=torch.cat([old,meta.input_ids.to(old.dtype)])
            owner.original_ngram(meta,None,context_pool)
            indices=fla.track_boundary_row[:,None]+torch.arange(old.numel(),device=old.device)[None,:]
            context_pool.index_copy_(0,fla.track_dst,history[indices])

        def update_index(backend,index,md,slot):
            fla=get_global_ctx().batch.fla_metadata
            if not getattr(fla,'_mtp_row_snapshots',False):
                return owner.original_index(index,md,slot)
            ring=backend.kvcache.pending_ring(slot)
            old=ring[owner.req.table_idx].clone()
            owner.original_index(index,md,slot)
            count=index.k.shape[0]
            owner.rings[:count,slot].copy_(old.unsqueeze(0).expand(count,-1,-1))
            for row in range(count):
                indices=md.positions[:row+1].long()%old.shape[0]
                owner.rings[row,slot].index_copy_(0,indices,index.k[:row+1])

        self.gdn_class._gguf_core=gdn_core
        self.ple_class._write_track_snapshot=ple_snapshot
        ple.commit_ngram_context=ngram
        backend._update_index_cache=types.MethodType(update_index,backend)

    def bind(self,batch):
        """用现有GGUF GDN任意行track机制保存recurrent，再补足早期conv/PLE历史。"""
        fla=batch.fla_metadata
        count=batch.input_ids.numel()
        fla._mtp_row_snapshots=True
        fla.track_dst=self.dst[:count]
        fla.track_boundary_row=torch.arange(1,count+1,device=self.engine.device,dtype=torch.int64)
        km1=self.pool.conv_states.shape[-1]
        fla.track_conv_src=torch.zeros((count,km1),device=self.engine.device,dtype=torch.int64)

    def restore_prefix(self,row):
        """选择已验证最后一行的完整语义状态，不再重新计算accepted前缀。"""
        self.pool.copy_from(self.slots[row],self.live)
        self.engine.kv_cache._pending_ring[self.req.table_idx].copy_(self.rings[row])

    def close(self):
        """在图已释放后还原诊断hooks并归还临时lease，保留正常prefix槽所有权。"""
        self.gdn_class._gguf_core=self.original_gdn
        self.ple_class._write_track_snapshot=self.original_ple
        self.ple_module.commit_ngram_context=self.original_ngram
        self.engine.attn_backend._update_index_cache=self.original_index
        self.pool.free(self.slots)
