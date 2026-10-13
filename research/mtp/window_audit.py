"""Diagnostic replay versus the unmodified single-token FreeToken decode graph."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from freetoken.core import Batch
from target_window import TargetWindow
from speculative_loop import state_refs


class AuditedWindow(TargetWindow):
    output=Path('mtp-window-native-audit-v1.json')
    records=[]

    def __init__(self,*args,**kwargs):
        """只用于诊断，不把重复计算和强制同步纳入性能结果。"""
        self.snapshots=kwargs.get('snapshots')
        super().__init__(*args,**kwargs)

    def replay(self,history,inputs,position):
        """同一真实前缀分别执行窗口和原decode图，首次差异立即保留状态幅度。"""
        refs=state_refs(self.engine,self.req)
        initial={name:t.clone() for name,t in refs.items()}
        result=super().replay(history,inputs,position)
        picks=result.cpu().tolist()
        raw=self.engine.model.model.hyper_connection_mixer._mtp_residual_capture
        raw_before=raw[:len(inputs)].clone()
        final={name:t.clone() for name,t in refs.items()}
        for name,t in refs.items(): t.copy_(initial[name])
        serial=[]
        errors=[]
        pool=self.engine.linear_state_pool
        for row,token in enumerate(inputs):
            req=copy.copy(self.req)
            req.cached_len=position+row
            req.device_len=position+row+1
            req.input_ids=torch.tensor(history+inputs[:row+1],dtype=torch.int32)
            batch=Batch(reqs=[req],phase='decode')
            batch.padded_reqs=[req]
            batch.input_ids=torch.tensor([token],device=self.engine.device,dtype=torch.int32)
            batch.positions=torch.tensor([position+row],device=self.engine.device,dtype=torch.int32)
            batch.out_loc=self.engine.page_table[req.table_idx,batch.positions.long()]
            slot=req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            batch.linear_table_idx=torch.tensor([slot],device=self.engine.device,dtype=torch.int32)
            batch.active_table_idx=torch.tensor([req.table_idx],device=self.engine.device,dtype=torch.int32)
            self.engine.attn_backend.prepare_metadata(batch)
            with self.engine.ctx.forward_batch(batch),self.engine.model.forward_host_ctx(batch,True):
                logits=self.engine.graph_runner.replay(batch)
            serial.append(int(logits.argmax(-1).item()))
            if self.snapshots is not None:
                target_slot=self.snapshots.slots[row]
                expected=dict(conv=pool.conv_states[:,target_slot],
                    recurrent=pool.recurrent_states[:,target_slot],qsa_pending=self.snapshots.rings[row])
                expected.update({name:t[:,target_slot] for name,t in pool.slot_states.items()})
                errors.append({name:float((t.float()-expected[name].float()).abs().max())
                    for name,t in refs.items()})
        record=dict(position=position,inputs=inputs,window_picks=picks,native_decode_picks=serial,
            per_row_state_max_abs_error=errors)
        self.records.append(record)
        self.output.write_text(json.dumps(self.records,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
        # A diagnostic abort avoids continuing with KV rewritten by the comparison.
        changed=picks!=serial or any(v!=0 for e in errors for v in e.values())
        for name,t in refs.items(): t.copy_(final[name])
        raw[:len(inputs)].copy_(raw_before)
        if changed: raise RuntimeError('Native decode/window difference recorded in '+str(self.output))
        return result
