#include "hybrid_q2_io_v1.h"
#include <freetoken/tensor.h>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <cuda_runtime.h>

namespace hybrid_q2_device_v1 {
// A single CTA copies the small quantized input and publishes its header last.
__global__ void publish(HybridQ2IoV1* io,const int8_t* q,const float* scale,const int32_t* ids,int layer,int tokens,int topk) {
    for(int i=threadIdx.x;i<tokens*2560;i+=blockDim.x)io->q[i]=q[i];
    for(int i=threadIdx.x;i<tokens*80;i+=blockDim.x)io->scale[i]=scale[i];
    for(int i=threadIdx.x;i<tokens*topk;i+=blockDim.x)io->ids[i]=ids[i];
    __threadfence_system();
    __syncthreads();
    if(threadIdx.x==0) {
        int count=0;for(int i=0;i<tokens*topk;++i)count+=ids[i]<0;
        io->layer=layer;io->tokens=tokens;io->topk=topk;io->cpu_count=count;
        if(count) {
            const uint64_t sequence=(*reinterpret_cast<volatile uint64_t*>(&io->request))+1;
            __threadfence_system();
            asm volatile("st.global.release.sys.u64 [%0], %1;"::"l"(&io->request),"l"(sequence):"memory");
        }
    }
}
__global__ void wait(HybridQ2IoV1* io) {
    if(threadIdx.x==0 && io->cpu_count) {
        const uint64_t wanted=(*reinterpret_cast<volatile uint64_t*>(&io->request));
        uint64_t observed;
        do {
            asm volatile("ld.global.acquire.sys.u64 %0, [%1];":"=l"(observed):"l"(&io->done):"memory");
            if(observed<wanted)__nanosleep(64);
        } while(observed<wanted);
    }
}
__global__ void merge(const HybridQ2IoV1* io,const int32_t* ids,float* parts,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && ids[i/2560]<0)parts[i]=io->parts[i];
}
}

struct HybridQ2DeviceV1 {
    static void publish(tvm::ffi::TensorView storage,tvm::ffi::TensorView q,
        tvm::ffi::TensorView scales,tvm::ffi::TensorView ids,int64_t layer,int64_t stream) {
        using namespace host;
        auto tokens=SymbolicSize{"tokens"},topk=SymbolicSize{"topk"};auto device=SymbolicDevice{};
        TensorMatcher({int64_t(sizeof(HybridQ2IoV1))}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(storage);
        TensorMatcher({tokens,2560}).with_dtype<int8_t>().with_device<kDLCUDA>(device).verify(q);
        TensorMatcher({tokens,80}).with_dtype<float>().with_device<kDLCUDA>(device).verify(scales);
        TensorMatcher({tokens,topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(ids);
        RuntimeCheck(tokens.unwrap()>=1 && tokens.unwrap()<=4 && topk.unwrap()==10 && layer>=0 && layer<48,"Invalid CPU handoff geometry");
        hybrid_q2_device_v1::publish<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<HybridQ2IoV1*>(storage.data_ptr()),static_cast<const int8_t*>(q.data_ptr()),
            static_cast<const float*>(scales.data_ptr()),static_cast<const int32_t*>(ids.data_ptr()),
            int(layer),int(tokens.unwrap()),int(topk.unwrap()));
        auto error=cudaGetLastError();RuntimeCheck(error==cudaSuccess,cudaGetErrorString(error));
    }
    static void collect(tvm::ffi::TensorView storage,tvm::ffi::TensorView parts,tvm::ffi::TensorView ids,int64_t stream) {
        using namespace host;
        auto routes=SymbolicSize{"routes"};auto device=SymbolicDevice{};
        TensorMatcher({int64_t(sizeof(HybridQ2IoV1))}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(storage);
        TensorMatcher({routes,2560}).with_dtype<float>().with_device<kDLCUDA>(device).verify(parts);
        RuntimeCheck(routes.unwrap()>=10 && routes.unwrap()<=40,"Invalid CPU output geometry");
        TensorMatcher({routes.unwrap()/10,10}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(ids);
        const auto cs=reinterpret_cast<cudaStream_t>(stream);
        auto* io=static_cast<HybridQ2IoV1*>(storage.data_ptr());
        hybrid_q2_device_v1::wait<<<1,32,0,cs>>>(io);
        const int count=int(routes.unwrap())*2560;
        hybrid_q2_device_v1::merge<<<(count+255)/256,256,0,cs>>>(io,static_cast<const int32_t*>(ids.data_ptr()),static_cast<float*>(parts.data_ptr()),count);
        auto error=cudaGetLastError();RuntimeCheck(error==cudaSuccess,cudaGetErrorString(error));
    }
};
