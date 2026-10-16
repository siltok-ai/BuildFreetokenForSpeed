"""诊断原型：运行时层偏移，保留所有准入/淘汰/统计表达式。"""
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["BASE"])
def _ensure_runtime_base(IDS, MAP, DYNAMIC_IDS, USAGE, STEP, COPY_SRC, COPY_DST, COUNT,
            STATS, BASE, DYNAMIC_BEGIN: tl.constexpr,
            PROTECTED_BEGIN: tl.constexpr, DYNAMIC_SIZE: tl.constexpr,
            K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
            COLLECT_STATS: tl.constexpr):
    step = tl.load(STEP) + 1
    tl.store(STEP, step)
    k = tl.arange(0, BK)
    valid = k < K
    ids = tl.load(IDS + k, valid, 0).to(tl.int64)
    global_ids = ids + BASE
    slots = tl.load(MAP + global_ids, valid, -1)
    hits = valid & (slots >= 0)
    dynamic_hits = hits & (slots >= DYNAMIC_BEGIN) & (slots < PROTECTED_BEGIN)
    tl.store(USAGE + slots - DYNAMIC_BEGIN, step, dynamic_hits)
    same = (global_ids[:, None] == global_ids[None, :]) & (k[:, None] > k[None, :]) & valid[:, None] & valid[None, :]
    first = valid & (tl.sum(same.to(tl.int32), 1) == 0)
    first_miss = first & (slots < 0)
    miss = valid & (slots < 0)
    rank = tl.sum(((global_ids[None, :] < global_ids[:, None]) & first_miss[None, :]).to(tl.int32), 1)
    n = tl.sum(first_miss.to(tl.int32))
    tl.store(COUNT, n.to(tl.int64))
    out = slots
    if n > 0:
        tl.debug_barrier()
        c = tl.arange(0, BC)
        u = tl.load(USAGE + c, c < DYNAMIC_SIZE, 0x3FFFFFFFFFFFFFFF)
        u = tl.where((c >= DYNAMIC_SIZE) | (u == step), 0x3FFFFFFFFFFFFFFF, u)
        for i in range(n):
            victim = tl.argmin(u, 0).to(tl.int32)
            old = tl.load(DYNAMIC_IDS + victim)
            if old >= 0:
                tl.store(MAP + old, -1)
            incoming = tl.sum(tl.where(first_miss & (rank == i), global_ids, 0), 0)
            physical = victim + DYNAMIC_BEGIN
            tl.store(DYNAMIC_IDS + victim, incoming.to(tl.int32))
            tl.store(MAP + incoming, physical)
            tl.store(USAGE + victim, step)
            tl.store(COPY_SRC + i, (incoming - BASE).to(tl.int32))
            tl.store(COPY_DST + i, physical)
            out = tl.where(miss & (rank == i), physical, out)
            u = tl.where(c == victim, 0x3FFFFFFFFFFFFFFF, u)
    tl.store(IDS + k, out, valid)
    if COLLECT_STATS:
        s = tl.arange(0, 4)
        values = tl.where(s == 0, tl.sum(first.to(tl.int32), 0), tl.where(s == 1, n, 1))
        tl.atomic_add(STATS + s, values.to(tl.int64), s < 3)


def enable():
    """仅替换当前测试进程的准入内核；源文件和普通wheel保持不变。"""
    import freetoken.moe.q2_residency as residency
    residency._ensure=_ensure_runtime_base
