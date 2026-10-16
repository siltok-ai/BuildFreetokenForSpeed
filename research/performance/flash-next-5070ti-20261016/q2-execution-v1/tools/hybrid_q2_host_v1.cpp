// Experimental mapped handoff to the unmodified Strata Q2 CPU kernels.
#define NOMINMAX
#include "hybrid_q2_io_v1.h"
#include "strata/kernels/cpu/expert.hpp"
#include <freetoken/tensor.h>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <windows.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace hybrid_q2_v1 {
using strata::kernels::cpu::ActQ;
using strata::kernels::cpu::act_quant_q8_1;
using strata::kernels::cpu::q2_0_gguf_rows_multi;

struct Group {int expert=0,count=0;int routes[4]={};};

struct Pool {
    HybridQ2IoV1* io;
    std::vector<int64_t> pointers;
    int threads;
    std::atomic<bool> stop{false};
    std::atomic<int> bad{0},epoch{0},next{0},remaining{0};
    int phase=0,jobs=0,ngroups=0,ncpu=0;
    std::array<Group,40> groups{};
    std::array<int,40> cpu_routes{};
    alignas(64) std::array<ActQ,4> a1{};
    alignas(64) std::array<ActQ,40> a2{};
    alignas(64) float gu[40][1280]{};
    alignas(64) float ff[40][640]{};
    std::vector<std::thread> workers;
    std::thread coordinator;
    std::condition_variable cv;
    std::mutex mutex;
    std::atomic<uint64_t> batches{0};

    Pool(HybridQ2IoV1* io_,const int64_t* source,int nt):io(io_),pointers(source,source+48*2*512),threads(nt) {
        for(int i=1;i<threads;++i) workers.emplace_back([this,i]{worker(i);});
        coordinator=std::thread([this]{coordinate();});
    }

    ~Pool() {
        stop.store(true,std::memory_order_release);
        cv.notify_all();
        coordinator.join();
        for(auto& worker:workers)worker.join();
    }

    // Each phase owns disjoint output rows. Only the coordinator publishes phases.
    void work() {
        while(true) {
            const int job=next.fetch_add(1,std::memory_order_relaxed);
            if(job>=jobs)break;
            if(phase==1 || phase==3) {
                const int row_count=phase==1?1280:2560;
                const int chunk=phase==1?16:32;
                const int per=row_count/chunk;
                const Group& group=groups[job/per];
                const int begin=(job%per)*chunk;
                const int role=phase==1?0:1;
                const auto ptr=pointers[(io->layer*2+role)*512+group.expert];
                if(!ptr) {bad.store(1);continue;}
                const ActQ* acts[4];float* out[4];
                for(int t=0;t<group.count;++t) {
                    const int route=group.routes[t];
                    acts[t]=phase==1?&a1[route/io->topk]:&a2[route];
                    out[t]=phase==1?gu[route]:io->parts+route*2560;
                }
                q2_0_gguf_rows_multi(reinterpret_cast<const uint8_t*>(ptr),phase==1?720:180,
                    phase==1?40:10,acts,group.count,out,begin,begin+chunk);
            } else {
                const int route=cpu_routes[job];
                for(int row=0;row<640;++row) {
                    const float g=gu[route][row];
                    ff[route][row]=(g/(1.0f+std::exp(-g)))*gu[route][640+row];
                }
                act_quant_q8_1(ff[route],640,a2[route]);
            }
        }
    }

    void run_phase(int p,int count) {
        phase=p;jobs=count;next.store(0,std::memory_order_relaxed);
        remaining.store(threads-1,std::memory_order_relaxed);
        epoch.fetch_add(1,std::memory_order_release);
        cv.notify_all();
        work();
        while(remaining.load(std::memory_order_acquire))_mm_pause();
    }

    // Short phase gaps spin; long prefill gaps park instead of occupying seven cores.
    void worker(int index) {
        SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<(2*index));
        int last=0;
        auto active=std::chrono::steady_clock::now();
        while(!stop.load(std::memory_order_acquire)) {
            const int observed=epoch.load(std::memory_order_acquire);
            if(observed==last) {
                _mm_pause();
                if(std::chrono::steady_clock::now()-active>std::chrono::milliseconds(50)) {
                    std::unique_lock lock(mutex);
                    cv.wait(lock,[&]{return stop.load() || epoch.load(std::memory_order_acquire)!=last;});
                }
                continue;
            }
            last=observed;
            work();
            remaining.fetch_sub(1,std::memory_order_release);
            active=std::chrono::steady_clock::now();
        }
    }

    void compute() {
        ncpu=0;ngroups=0;
        const int routes=io->tokens*io->topk;
        for(int token=0;token<io->tokens;++token) {
            std::memcpy(a1[token].q,io->q+token*2560,2560);
            std::memcpy(a1[token].scale,io->scale+token*80,80*sizeof(float));
            a1[token].nchunks=80;
            for(int c=0;c<80;++c) {
                int sum=0;
                for(int j=0;j<32;++j)sum+=int(a1[token].q[c*32+j]);
                a1[token].sum[c]=sum;a1[token].hx[c]=a1[token].scale[c]*float(sum);
            }
        }
        for(int route=0;route<routes;++route) {
            if(io->ids[route]>=0)continue;
            const int expert=-io->ids[route]-1;
            int group=0;
            for(;group<ngroups && groups[group].expert!=expert;++group){}
            if(group==ngroups){groups[group]=Group{};groups[group].expert=expert;++ngroups;}
            auto& g=groups[group];
            if(g.count==4 || expert<0 || expert>=512)throw std::runtime_error("Invalid CPU expert group");
            g.routes[g.count++]=route;cpu_routes[ncpu++]=route;
        }
        run_phase(1,ngroups*80);
        run_phase(2,ncpu);
        run_phase(3,ngroups*80);
    }

    // The GPU release-store publishes all input bytes; completion is a CPU release-store.
    void coordinate() {
        SetThreadAffinityMask(GetCurrentThread(),1);
        uint64_t last=0;
        while(!stop.load(std::memory_order_acquire)) {
            const uint64_t sequence=std::atomic_ref(io->request).load(std::memory_order_acquire);
            if(sequence==last){_mm_pause();continue;}
            try {compute();}
            catch(...) {bad.store(1);}
            last=sequence;batches.fetch_add(1,std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            std::atomic_ref(io->done).store(sequence,std::memory_order_release);
        }
    }
};
}

struct HybridQ2HostV1 {
    static int64_t io_bytes(){return sizeof(HybridQ2IoV1);}
    static int64_t create(tvm::ffi::TensorView io,tvm::ffi::TensorView pointers,int64_t threads) {
        using namespace host;
        TensorMatcher({int64_t(sizeof(HybridQ2IoV1))}).with_dtype<uint8_t>().with_device<kDLCPU>().verify(io);
        TensorMatcher({48,2,512}).with_dtype<int64_t>().with_device<kDLCPU>().verify(pointers);
        RuntimeCheck(threads>=1 && threads<=8,"CPU prototype supports1..8 workers");
        RuntimeCheck(strata::kernels::cpu::cpu_features().usable(),"CPU prototype needs AVX512 VNNI/VBMI/VL");
        std::memset(io.data_ptr(),0,sizeof(HybridQ2IoV1));
        return reinterpret_cast<int64_t>(new hybrid_q2_v1::Pool(static_cast<HybridQ2IoV1*>(io.data_ptr()),
            static_cast<const int64_t*>(pointers.data_ptr()),int(threads)));
    }
    static void update(int64_t handle,tvm::ffi::TensorView pointers) {
        using namespace host;
        TensorMatcher({48,2,512}).with_dtype<int64_t>().with_device<kDLCPU>().verify(pointers);
        auto* pool=reinterpret_cast<hybrid_q2_v1::Pool*>(handle);
        RuntimeCheck(std::atomic_ref(pool->io->done).load()==std::atomic_ref(pool->io->request).load(),"Drain CPU work before updating owners");
        std::memcpy(pool->pointers.data(),pointers.data_ptr(),48*2*512*sizeof(int64_t));
    }
    static int64_t health(int64_t handle){return reinterpret_cast<hybrid_q2_v1::Pool*>(handle)->bad.load();}
    static int64_t batches(int64_t handle){return reinterpret_cast<hybrid_q2_v1::Pool*>(handle)->batches.load();}
    static void close(int64_t handle){delete reinterpret_cast<hybrid_q2_v1::Pool*>(handle);}
};
