"""Single-request causal target graphs with moving GPU metadata; diagnostic integration only."""
from __future__ import annotations

import copy

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch


class TargetWindow:
    def __init__(self,engine,req,initial_position,*,max_rows=4):
        """预热并捕获每种窗口，使用causal元数据、all-row head及现有PLE graph staging。"""
        self.engine=engine
        self.req=req
        self.ple=engine.model._ple_table
        self.graphs={}
        import freetoken.models.qwen4_exp.gdn as gdn
        from freetoken.kernel.triton.causal_conv1d_triton import causal_conv1d_varlen
        original=gdn.causal_conv1d_varlen
        def static_conv(x,weight,states,cu,indices,initial):
            return causal_conv1d_varlen(x,weight,states,cu,indices,initial,
                max_seq_len=x.shape[-1],batch=cu.numel()-1)
        gdn.causal_conv1d_varlen=static_conv
        try:
            for count in range(1,max_rows+1):
                virtual=copy.copy(req)
                virtual.cached_len=initial_position
                virtual.device_len=initial_position+count
                virtual.mamba_ping_pong=None
                virtual.input_ids=torch.cat([req.input_ids[:initial_position],torch.zeros(count,dtype=torch.int32)])
                batch=Batch(reqs=[virtual],phase='prefill')
                batch.padded_reqs=[virtual]
                batch.input_ids=torch.zeros(count,device=engine.device,dtype=torch.int32)
                batch.positions=torch.arange(initial_position,initial_position+count,device=engine.device,dtype=torch.int32)
                batch.out_loc=engine.page_table[req.table_idx,batch.positions.long()].clone()
                batch.fla_metadata=build_fla_metadata(batch,engine.device)
                engine.attn_backend.prepare_metadata(batch)
                for _ in range(2):
                    with engine.ctx.forward_batch(batch),engine.model.forward_host_ctx(batch,False):
                        mixed=engine.model.model.forward(batch.input_ids,batch)
                        engine.model.lm_head.quant_method.apply(engine.model.lm_head,mixed)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=engine.stream):
                    with engine.ctx.forward_batch(batch):
                        mixed=engine.model.model.forward(batch.input_ids,batch)
                        logits=engine.model.lm_head.quant_method.apply(engine.model.lm_head,mixed)
                        picks=logits.argmax(-1).to(torch.int32)
                host=torch.empty(count,dtype=torch.int32,pin_memory=True)
                self.graphs[count]=(batch,graph,picks,host)
        finally:
            gdn.causal_conv1d_varlen=original

    def replay(self,history,inputs,position):
        """更新原地址的输入/位置/长度/分页映射，再正确填多行PLE并重放。"""
        batch,graph,picks,host=self.graphs[len(inputs)]
        host.copy_(torch.tensor(inputs,dtype=torch.int32))
        batch.input_ids.copy_(host,non_blocking=True)
        batch.positions.copy_(torch.arange(position,position+len(inputs),device=self.engine.device,dtype=torch.int32))
        batch.out_loc.copy_(self.engine.page_table[self.req.table_idx,batch.positions.long()])
        md=batch.attn_metadata
        md.seq_lens.fill_(position+len(inputs))
        md.kv_len_cpu.fill_(position+len(inputs))
        md.block_table.copy_(self.engine.attn_backend._block_table(md.ring_slots.long()))
        cpu_ids=torch.tensor(history+list(inputs),dtype=torch.int32)
        context=self.ple._ple_context(cpu_ids,position)
        run=torch.tensor([*context,*inputs],dtype=torch.int64)
        self.ple.fill([run],graph=True)
        graph.replay()
        self.ple._graph_consumed.record(self.engine.stream)
        return picks

    def close(self):
        """释放独立诊断图；原FreeToken单token图及底层所有者保持不变。"""
        torch.cuda.synchronize()
        for _,graph,_,_ in self.graphs.values(): graph.reset()
        self.graphs.clear()
