#pragma once
#include <cstdint>

// One outstanding layer per model stream. The mapped owner outlives all graphs.
struct alignas(64) HybridQ2IoV1 {
    alignas(64) uint64_t request=0;
    alignas(64) uint64_t done=0;
    alignas(64) int layer=0,tokens=0,topk=0,cpu_count=0;
    int ids[40]={};
    alignas(64) int8_t q[4*2560]={};
    alignas(64) float scale[4*80]={};
    alignas(64) float parts[40*2560]={};
};
