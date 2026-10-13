"""Isolated greedy speculative loop on an owned FreeToken request; not a production scheduler."""
from __future__ import annotations

import copy
import time

import torch
from target_window import TargetWindow


def state_refs(engine,req):
    """单请求事务包括所有GDN/PLE状态与QSA跨调用ring，保留实际tensor布局。"""
    pool=engine.linear_state_pool
    slot=req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
    refs=dict(conv=pool.conv_states[:,slot],recurrent=pool.recurrent_states[:,slot],
        qsa_pending=engine.kv_cache._pending_ring[req.table_idx])
    refs.update({name:t[:,slot] for name,t in pool.slot_states.items()})
    return refs


def run_speculative(engine,cache_manager,req,draft,prompt_ids,pending,raw_last,*,max_tokens,eos_ids,direct_snapshots=False,decode_conv=False,draft_graphs=False,scalar_projections=False,decode_ple=False):
    """预留独占诊断页和事务buffer，逐轮验证/提交，结束时恢复调用方全部所有权。"""
    if max_tokens<1: raise ValueError('Speculative output limit must be positive')
    refs=state_refs(engine,req)
    initial={name:t.clone() for name,t in refs.items()}
    backup={name:torch.empty_like(t) for name,t in refs.items()}
    free_before=cache_manager.free_slots.clone()
    table_before=engine.page_table[req.table_idx].clone()
    position=req.cached_len
    history=list(prompt_ids)
    assert len(history)==position
    if len(cache_manager.free_slots)<(max_tokens+7)//engine.ctx.page_size+2:
        raise RuntimeError('Diagnostic request requires enough free pages without prefix eviction')
    reserve=copy.copy(req)
    reserve.cached_len=position
    reserve.device_len=position+max_tokens+4
    cache_manager.allocate_paged([reserve])
    graphs=None
    snapshots=None
    draft_runner=None
    def restore(snapshot):
        for name,t in refs.items(): t.copy_(snapshot[name])
    try:
        started=time.perf_counter()
        if direct_snapshots:
            from row_snapshots import RowSnapshots
            snapshots=RowSnapshots(engine,req)
        graphs=TargetWindow(engine,req,position,snapshots=snapshots,decode_conv=decode_conv,scalar_projections=scalar_projections,decode_ple=decode_ple)
        if draft_graphs:
            from draft_graph import DraftGraphs
            width=min(draft.max_context,1<<(position+max_tokens+3-1).bit_length())
            draft_runner=DraftGraphs(draft,engine.stream,attention_width=width)
        restore(initial)
        torch.cuda.synchronize()
        prepare_seconds=time.perf_counter()-started
        output=[int(pending)]
        arrivals=[time.perf_counter()]
        rounds=[]
        raw=raw_last
        raw_capture=engine.model.model.hyper_connection_mixer._mtp_residual_capture
        while len(output)<max_tokens and pending not in eos_ids:
            remaining=max_tokens-len(output)
            depth=min(3,remaining-1)
            candidates=[]
            token=torch.tensor([pending],device=engine.device,dtype=torch.int32)
            r=raw
            before=time.perf_counter()
            if draft_runner is not None and depth:
                candidates=draft_runner.chain(raw,pending,position-1,depth)
                first_eos=next((i for i,t in enumerate(candidates) if t in eos_ids),None)
                if first_eos is not None: candidates=candidates[:first_eos+1]
            else:
                for step in range(depth):
                    token,r=draft.step(r,token,position-1+step)
                    candidate=int(token.item())
                    candidates.append(candidate)
                    if candidate in eos_ids: break
            after_draft=time.perf_counter()
            if snapshots is None:
                for name,t in refs.items(): backup[name].copy_(t)
            inputs=[pending,*candidates]
            picks=graphs.replay(history,inputs,position).cpu().tolist()
            after_verify=time.perf_counter()
            accepted=0
            for actual,want in zip(candidates,picks):
                if actual!=want or actual in eos_ids: break
                accepted+=1
            commit_rows=accepted+1
            if commit_rows<len(inputs):
                if snapshots is None:
                    restore(backup)
                    graphs.replay(history,inputs[:commit_rows],position)
                else:
                    snapshots.restore_prefix(commit_rows-1)
            committed_raw=raw_capture[:commit_rows].clone()
            # 把草稿以前用自身R生成的中间KV换成已验证target R；最后cell留给下一轮draft step。
            if accepted:
                catchup_tokens=torch.tensor(picks[:accepted],device=engine.device,dtype=torch.int32)
                if draft_runner is None:
                    draft.prime(committed_raw[:accepted],catchup_tokens,start=position)
                else:
                    draft_runner.prime(committed_raw[:accepted],catchup_tokens,position)
            raw=committed_raw[-1:]
            history.extend(inputs[:commit_rows])
            position+=commit_rows
            pending=int(picks[accepted])
            delivered=candidates[:accepted]+[pending]
            torch.cuda.synchronize()
            at=time.perf_counter()
            output.extend(delivered)
            arrivals.extend([at]*len(delivered))
            rounds.append(dict(input_rows=len(inputs),accepted_drafts=accepted,
                committed_rows=commit_rows,delivered_ids=delivered,
                draft_ms=1000*(after_draft-before),verify_ms=1000*(after_verify-after_draft),
                commit_and_catchup_ms=1000*(at-after_verify)))
        visible=output[:-1] if output[-1] in eos_ids else output
        times=arrivals[:len(visible)]
        return dict(scope='Actual isolated speculative token loop after graph preparation; no IPC or Speed acceptance.',
            graph_prepare_seconds=prepare_seconds,output_ids=visible,
            finish_reason='stop' if output[-1] in eos_ids else 'length',
            output_tokens=len(visible),decode_tps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
            arrival_offsets=[t-times[0] for t in times],rounds=rounds,
            draft_count=sum(r['input_rows']-1 for r in rounds),
            accepted_drafts=sum(r['accepted_drafts'] for r in rounds),
            target_window_count=len(rounds),direct_row_snapshots=direct_snapshots,decode_conv=decode_conv,draft_graphs=draft_graphs,scalar_projections=scalar_projections,decode_ple=decode_ple,
            partial_windows=sum(r['committed_rows']<r['input_rows'] for r in rounds),
            replayed_windows=0 if direct_snapshots else sum(r['committed_rows']<r['input_rows'] for r in rounds))
    finally:
        restore(initial)
        engine.page_table[req.table_idx].copy_(table_before)
        cache_manager.free_slots=free_before
        if graphs is not None: graphs.close()
        if draft_runner is not None: draft_runner.close()
        if snapshots is not None: snapshots.close()
        torch.cuda.synchronize()
