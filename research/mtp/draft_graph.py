"""Owned greedy draft-chain graphs with GPU positions and FreeToken paged attention."""
from __future__ import annotations

import torch
import types
from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention


class _StageOneKernel:
    def __init__(self,kernel): self.kernel=kernel
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            kwargs['num_stages']=1
            return self.kernel[grid](*args,**kwargs)
        return launch


class DraftGraphs:
    def __init__(self,draft,stream,*,attention_width):
        """捕获固定1/2/3草稿链和有效前缀追齐图；复用原压缩专家及FP32 KV。"""
        if not 0<attention_width<=draft.max_context or draft.max_context%64:
            raise ValueError('Draft graph requires a bounded page-aligned context')
        self.draft=draft
        self.stream=stream
        device=draft.device
        self.cell=torch.zeros(1,device=device,dtype=torch.int32)
        self.r=torch.zeros((1,10240),device=device,dtype=torch.float32)
        self.token=torch.zeros(1,device=device,dtype=torch.int32)
        self.candidates=torch.empty(3,device=device,dtype=torch.int32)
        self.columns=torch.arange(attention_width,device=device,dtype=torch.int32)[None,:]
        self.table=torch.arange(draft.max_context//64,device=device,dtype=torch.int32)[None,:]
        self.mapping=torch.zeros(1,device=device,dtype=torch.int32)
        self.keys=draft.keys.view(-1,64,2,256)
        self.values=draft.values.view(-1,64,2,256)
        self.catchup_r=torch.zeros((3,10240),device=device,dtype=torch.float32)
        self.catchup_tokens=torch.zeros(3,device=device,dtype=torch.int32)
        self.catchup_start=torch.zeros(1,device=device,dtype=torch.int32)
        self.chains={}
        self.catchup={}
        self.attention_width=attention_width
        # FP32 K/V with the 12-query-head group exceeds SM120 shared memory at
        # Triton's default pipeline depth. Fork only the wrapper's globals;
        # the main FreeToken kernel and installed module remain unchanged.
        globals_copy=dict(qsa_sparse_paged_attention.__globals__)
        globals_copy['_qsa_sparse_paged_gqa_splitk_kernel']=_StageOneKernel(
            globals_copy['_qsa_sparse_paged_gqa_splitk_kernel'])
        self.attention=types.FunctionType(qsa_sparse_paged_attention.__code__,globals_copy,
            qsa_sparse_paged_attention.__name__,qsa_sparse_paged_attention.__defaults__)
        # Capture touches only early KV cells; restore them before request execution.
        key_before=draft.keys[:3].clone()
        value_before=draft.values[:3].clone()
        try:
            for count in range(1,4):
                def chain():
                    for row in range(count):
                        token,residual=draft.step(self.r,self.token,0,
                            positions=self.cell,attend=self.attend)
                        self.r.copy_(residual)
                        self.token.copy_(token)
                        self.candidates[row].copy_(token[0])
                        self.cell.add_(1)
                self.cell.zero_()
                chain()
                self.cell.zero_()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): chain()
                self.chains[count]=graph
                def catchup():
                    r=draft.initial_residual(self.catchup_r[:count],self.catchup_tokens[:count])
                    x,_=draft.mix('attn_hyper_connection',r)
                    positions=self.catchup_start+torch.arange(count,device=device,dtype=torch.int32)
                    draft.key_value(x,positions)
                catchup()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): catchup()
                self.catchup[count]=graph
        finally:
            draft.keys[:3].copy_(key_before)
            draft.values[:3].copy_(value_before)
            torch.cuda.synchronize()

    def attend(self,q,positions):
        """用GPU cell掩掉未提交/未来KV，FP32分页GQA不要求CPU读取长度。"""
        indices=torch.where(self.columns<=positions[0],self.columns,-1).to(torch.int32)
        return self.attention(q.reshape(1,24,256),self.keys,self.values,
            indices,self.table,self.mapping).reshape(1,24,1,256)

    def chain(self,residual,token,cell,count):
        """一次提交整条草稿链，只在返回完整候选时同步CPU。"""
        if count not in self.chains or cell<0 or cell+count>self.attention_width:
            raise ValueError('Draft chain is outside the captured attention capacity')
        self.r.copy_(residual)
        self.token.fill_(int(token))
        self.cell.fill_(cell)
        self.chains[count].replay()
        return self.candidates[:count].cpu().tolist()

    def prime(self,residuals,tokens,start):
        """只追齐已经接受的主R，不把拒绝尾部作为后续草稿上下文。"""
        count=len(tokens)
        if count not in self.catchup or start+count>self.attention_width:
            raise ValueError('Draft catch-up is outside the captured capacity')
        self.catchup_r[:count].copy_(residuals)
        self.catchup_tokens[:count].copy_(tokens)
        self.catchup_start.fill_(start)
        self.catchup[count].replay()

    def close(self):
        """在完成所有读写后释放私有图，不接管主模型任何权重或KV所有权。"""
        torch.cuda.synchronize()
        for graph in [*self.chains.values(),*self.catchup.values()]: graph.reset()
        self.chains.clear()
        self.catchup.clear()
