// Prototype: reuse each original Q2 block across up to four routed tokens.
// Activations retain the downstream FP32 scales, not native Q8_1 half scales.
#include <freetoken/tensor.h>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace grouped_q2_proto2 {
template<int NC, int LANES>
__global__ void project(const int32_t* q, const float* scales, const uint8_t* weights,
                       const int32_t* ids, const int32_t* meta, float* output,
                       int k, int n, int topk, int input_topk, int routes) {
    const int route=blockIdx.y;
    if (meta[route*(NC+1)]!=route) return;
    const int row=blockIdx.x*(256/LANES)+threadIdx.x/LANES;
    const int lane=threadIdx.x%LANES;
    if (row>=n) return;
    const int expert=ids[route];
    const int groups=k/32;
    const uint8_t* w=weights+(int64_t(expert)*n+row)*(k/64*18);
    int target[NC];
    float acc[NC]={};
    #pragma unroll
    for(int c=0;c<NC;++c) target[c]=meta[route*(NC+1)+1+c];
    for(int group=lane;group<groups;group+=LANES) {
        const uint8_t* block=w+(group/2)*18;
        const uint16_t raw=uint16_t(block[0])|(uint16_t(block[1])<<8);
        const float dw=__half2float(__ushort_as_half(raw));
        int packed[8];
        #pragma unroll
        for(int j=0;j<8;++j) {
            const unsigned code=block[2+(group%2)*8+j];
            packed[j]=int((code&3)|(((code>>2)&3)<<8)|(((code>>4)&3)<<16)|(((code>>6)&3)<<24));
        }
        #pragma unroll
        for(int c=0;c<NC;++c) {
            if(target[c]<routes) {
                const int input=target[c]/input_topk;
                const int32_t* x=q+input*(k/4)+group*8;
                int dot=0;
                #pragma unroll
                for(int j=0;j<8;++j) dot+=__dp4a(packed[j],x[j],0)-__dp4a(0x01010101,x[j],0);
                const float term=__fmul_rn(__fmul_rn(float(dot),dw),scales[input*groups+group]);
                acc[c]=__fadd_rn(acc[c],term);
            }
        }
    }
    #pragma unroll
    for(int c=0;c<NC;++c) {
        #pragma unroll
        for(int shift=LANES/2;shift;shift/=2) acc[c]=__fadd_rn(acc[c],__shfl_xor_sync(__activemask(),acc[c],shift,LANES));
        if(lane==0 && target[c]<routes) output[int64_t(target[c])*n+row]=acc[c];
    }
}
}

struct NativeGroupedQ2Proto2 {
    // Check the small-window layout at the FFI boundary; pointers never own data.
    static void project(tvm::ffi::TensorView q, tvm::ffi::TensorView scales,
                        tvm::ffi::TensorView weights, tvm::ffi::TensorView ids,
                        tvm::ffi::TensorView meta, tvm::ffi::TensorView output,
                        int64_t input_topk, int64_t lanes, int64_t stream) {
        using namespace host;
        auto tokens=SymbolicSize{"tokens"}, topk=SymbolicSize{"topk"};
        auto inputs=SymbolicSize{"inputs"}, words=SymbolicSize{"words"};
        auto slots=SymbolicSize{"slots"}, rows=SymbolicSize{"rows"}, bytes=SymbolicSize{"bytes"};
        auto device=SymbolicDevice{};
        TensorMatcher({tokens,topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(ids);
        TensorMatcher({inputs,words}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(q);
        const int t=int(tokens.unwrap()), r=t*int(topk.unwrap()), k=int(words.unwrap())*4;
        RuntimeCheck(t>=2 && t<=4 && k%64==0,"Invalid Q2 small window");
        RuntimeCheck(input_topk==1 || input_topk==topk.unwrap(),"Invalid Q2 input routing");
        RuntimeCheck(inputs.unwrap()==r/input_topk,"Invalid Q2 input rows");
        TensorMatcher({inputs,k/32}).with_dtype<float>().with_device<kDLCUDA>(device).verify(scales);
        TensorMatcher({slots,rows,bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(weights);
        RuntimeCheck(bytes.unwrap()==k/64*18,"Invalid Q2 compressed row");
        TensorMatcher({r,rows}).with_dtype<float>().with_device<kDLCUDA>(device).verify(output);
        const int nc=t==2?2:4, n=int(rows.unwrap());
        TensorMatcher({r,nc+1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(meta);
        RuntimeCheck(lanes==4 || lanes==16 || lanes==32,"Invalid Q2 subwarp");
        const auto launch=[&]<int NC,int L>() {
            grouped_q2_proto2::project<NC,L><<<dim3((n+256/L-1)/(256/L),r),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
                static_cast<const int32_t*>(q.data_ptr()),static_cast<const float*>(scales.data_ptr()),
                static_cast<const uint8_t*>(weights.data_ptr()),static_cast<const int32_t*>(ids.data_ptr()),
                static_cast<const int32_t*>(meta.data_ptr()),static_cast<float*>(output.data_ptr()),
                k,n,int(topk.unwrap()),int(input_topk),r);
        };
        if(nc==2) { if(lanes==4)launch.template operator()<2,4>();else if(lanes==16)launch.template operator()<2,16>();else launch.template operator()<2,32>(); }
        else { if(lanes==4)launch.template operator()<4,4>();else if(lanes==16)launch.template operator()<4,16>();else launch.template operator()<4,32>(); }
        const auto error=cudaGetLastError();
        RuntimeCheck(error==cudaSuccess,cudaGetErrorString(error));
    }
};
