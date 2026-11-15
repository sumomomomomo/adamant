#include "basalt/models/qwen4exp/prefill/fused_peer.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/ggml/ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
namespace f = basalt::prefill::fused;
constexpr int H = 2560, FF = 640, E = 32, K = 10;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc(&p, n)); }
    ~Buffer() { cudaFree(p); }
};
void fill(uint8_t* p, ggml_type type, int rows, int cols, std::mt19937& rng) {
    const size_t block = ggml_type_size(type), count = (size_t) rows * (cols / ggml_blck_size(type));
    for (size_t i = 0; i < count * block; ++i) p[i] = (uint8_t) rng();
    for (size_t i = 0; i < count; ++i) {
        const auto scale = ggml_fp32_to_fp16((type == GGML_TYPE_IQ4_XS ? 0.00005f : 0.0002f) * (1 + rng() % 6));
        std::memcpy(p + i * block, &scale, sizeof scale);
    }
}
void run(ggml_type gu, ggml_type down) {
    std::mt19937 rng(71);
    const size_t gr = ggml_row_size(gu, H), dr = ggml_row_size(down, FF);
    f::NativeGeom g{gu, down, gr, dr, gr * FF, gr * FF * 2, gu, gr};
    const size_t bytes = g.down_off + dr * H;
    std::vector<void*> primary(E), peer(E);
    for (int e = 0; e < E; ++e) {
        std::vector<uint8_t> data(bytes);
        fill(data.data(), gu, FF * 2, H, rng); fill(data.data() + g.down_off, down, H, FF, rng);
        ck(cudaMalloc(&primary[e], bytes)); ck(cudaMemcpy(primary[e], data.data(), bytes, cudaMemcpyHostToDevice));
        { const basalt::core::OnDevice on(1); ck(cudaMalloc(&peer[e], bytes)); ck(cudaMemcpy(peer[e], data.data(), bytes, cudaMemcpyHostToDevice)); }
    }
    basalt::prefill::FusedPeer helper;
    std::string error;
    if (!helper.open(0, 1, 16384, E, error)) throw std::runtime_error(error);
    cudaStream_t cs; ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    for (int t : {33, 257, 4096, 16384}) {
        const int64_t rows = (int64_t) t * K;
        std::vector<float> x((size_t) t * H);
        std::normal_distribution<float> normal;
        for (float& v : x) v = normal(rng);
        std::fill(x.begin(), x.begin() + H, 0.f);
        std::vector<int32_t> ids(rows);
        for (int i = 0; i < t; ++i) for (int k = 0; k < K; ++k) ids[(size_t) i * K + k] = (i % (E - K - 2) + k);
        Buffer dx(x.size()*4), di(ids.size()*4), xa(f::act_bytes(t,H)), grouping(f::group_bytes(rows,E));
        Buffer slot(rows*4), source(rows*4), ha(f::act_bytes(rows,FF)), output((size_t) rows*H*4);
        ck(cudaMemcpyAsync(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,cs));
        ck(cudaMemcpyAsync(di.p,ids.data(),ids.size()*4,cudaMemcpyHostToDevice,cs));
        f::quantize_act_native((float*) dx.p,t,H,xa.p,cs);
        f::group((int32_t*)di.p,rows,K,E,grouping.p,(int32_t*)slot.p,(int32_t*)source.p,cs);
        f::Batch all; all.e0=0; all.e1=E;
        for (int e=0;e<E;++e) all.blob[e]=(uint8_t*)primary[e];
        f::experts_native(all,g,E,rows,grouping.p,xa.p,(int32_t*)source.p,ha.p,(float*)output.p,cs);
        std::vector<float> expected((size_t)rows*H), actual(expected.size());
        ck(cudaMemcpyAsync(expected.data(),output.p,expected.size()*4,cudaMemcpyDeviceToHost,cs)); ck(cudaStreamSynchronize(cs));
        for (int mode : {0,1,2}) {
            ck(cudaMemsetAsync(output.p,0xff,expected.size()*4,cs));
            std::vector<f::Batch> batches;
            for (int e=0;e<E;++e) if (mode==2 || (mode==0 && e%2==0)) {
                f::Batch b; b.e0=e; b.e1=e+1; b.blob[0]=(uint8_t*)peer[e]; batches.push_back(b);
            }
            const auto begin=std::chrono::steady_clock::now();
            if (!helper.start(batches,g,t,xa.p,grouping.p,(int32_t*)source.p,(float*)output.p,cs,error)) throw std::runtime_error(error);
            for (int e=0;e<E;++e) if (mode==1 || (mode==0 && e%2!=0)) {
                f::Batch b; b.e0=e; b.e1=e+1; b.blob[0]=(uint8_t*)primary[e];
                f::experts_native(b,g,E,rows,grouping.p,xa.p,(int32_t*)source.p,ha.p,(float*)output.p,cs);
            }
            if (!helper.wait(cs,error)) throw std::runtime_error(error);
            ck(cudaMemcpyAsync(actual.data(),output.p,actual.size()*4,cudaMemcpyDeviceToHost,cs)); ck(cudaStreamSynchronize(cs));
            if (std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))) throw std::runtime_error("expert row bytes differ");
            for (float value : actual) if (!std::isfinite(value)) throw std::runtime_error("expert row is not finite");
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            std::printf("gu=%d down=%d tokens=%d ownership=%d returned=%lld: bitwise PASS, %.2f ms including download\n",gu,down,t,mode,(long long)helper.rows(),ms);
        }
    }
    ck(cudaStreamDestroy(cs));
    for (void* p:primary) cudaFree(p);
    {const basalt::core::OnDevice on(1);for(void*p:peer)cudaFree(p);}
}
}
int main() {
    int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count<2) return 77;
    try { run(GGML_TYPE_IQ4_XS,GGML_TYPE_IQ4_NL); run(GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_NL); run(GGML_TYPE_IQ2_S,GGML_TYPE_IQ4_NL); }
    catch(const std::exception&e){std::fprintf(stderr,"prefill_fused_peer_test: %s\n",e.what());return 1;}
    std::puts("prefill_fused_peer_test: PASS");
}
