"""Q2 owner pointers and row strides are 16-byte aligned; keep every copied byte."""
import triton
import triton.language as tl

@triton.jit
def _owner_gather_aligned(DST, SOURCES, FEATS, DI, SI, COUNT,
                  TABLE_SIZE: tl.constexpr, BLOCKS: tl.constexpr,
                  UNITS: tl.constexpr):
    bank = tl.program_id(0)
    block = tl.program_id(1)
    count = tl.load(COUNT)
    units_per_row = tl.load(FEATS + bank) // 16
    destination = tl.multiple_of(tl.load(DST + bank), 16)
    total = count * units_per_row
    lanes = tl.arange(0, UNITS)
    words = tl.arange(0, 4)
    for start in range(0, tl.cdiv(total, BLOCKS * UNITS)):
        unit = start * (BLOCKS * UNITS) + block * UNITS + lanes
        valid = unit < total
        route = unit // units_per_row
        src_id = tl.load(SI + route, valid, 0).to(tl.int64)
        dst_id = tl.load(DI + route, valid, 0).to(tl.int64)
        source = tl.multiple_of(tl.load(SOURCES + bank * TABLE_SIZE + src_id, valid, 0), 16)
        col = (unit % units_per_row) * 16
        src_addr = (source + col).to(tl.pointer_type(tl.uint32))
        dst_addr = (destination + dst_id * units_per_row * 16 + col).to(tl.pointer_type(tl.uint32))
        value = tl.load(src_addr[:, None] + words[None, :], valid[:, None], 0)
        tl.store(dst_addr[:, None] + words[None, :], value, valid[:, None])


def enable():
    """Replace only the protected Q2 gather kernel in the current process."""
    import freetoken.moe.q2_residency as module
    module._owner_gather = _owner_gather_aligned
