"""复用已在GPU动态区的incoming专家；outgoing仍回到原host槽，不增加RAM。"""
import triton
import triton.language as tl


@triton.jit
def exchange(GPU,HOST,FEATS,SLOTS,COUNT,COMMANDS,MAP,STATS,
    MAX_SWAPS:tl.constexpr,BLOCKS:tl.constexpr,UNITS:tl.constexpr,
    EXPERTS:tl.constexpr,PROTECTED_BEGIN:tl.constexpr):
    bank=tl.program_id(0);block=tl.program_id(1)
    count=tl.load(COUNT);row_units=tl.load(FEATS+bank)//16
    gpu=tl.load(GPU+bank)
    lanes=tl.arange(0,UNITS);words=tl.arange(0,4)
    total=count*row_units
    for start in range(tl.cdiv(total,BLOCKS*UNITS)):
        unit=start*(BLOCKS*UNITS)+block*UNITS+lanes
        valid=unit<total;item=unit//row_units
        slot=tl.load(SLOTS+item,valid,0).to(tl.int64)
        host=tl.load(HOST+bank*MAX_SWAPS+item,valid,0)
        layer=tl.load(COMMANDS+item*4,valid,0).to(tl.int64)
        cold=tl.load(COMMANDS+item*4+1,valid,0).to(tl.int64)
        previous=tl.load(MAP+layer*EXPERTS+cold,valid,-1).to(tl.int64)
        cached=(previous>=0)&(previous<PROTECTED_BEGIN)
        column=(unit%row_units)*16
        destination=(gpu+slot*row_units*16+column).to(tl.pointer_type(tl.uint32))
        host_ptr=(host+column).to(tl.pointer_type(tl.uint32))
        incoming=tl.where(cached,gpu+previous*row_units*16+column,host+column).to(tl.pointer_type(tl.uint32))
        old=tl.load(destination[:,None]+words[None,:],valid[:,None],0)
        new=tl.load(incoming[:,None]+words[None,:],valid[:,None],0)
        tl.store(destination[:,None]+words[None,:],new,valid[:,None])
        tl.store(host_ptr[:,None]+words[None,:],old,valid[:,None])
        if bank==0:
            # Count one row start, never all of its chunks; diagnostics read this after requests.
            first=valid&(column==0)
            tl.atomic_add(STATS+tl.where(cached,0,1),1,first)
