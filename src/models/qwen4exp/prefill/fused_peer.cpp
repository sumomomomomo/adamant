#include "basalt/models/qwen4exp/prefill/fused_peer.hpp"
#include "basalt/core/host_plan.hpp"
#include "basalt/core/on_device.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <thread>

namespace basalt::prefill {
namespace {
constexpr int H = 2560, FF = 640, K = 10;
size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class Allocate> void host_layout(int64_t t, int e, Allocate allocate) {
    allocate(fused::act_bytes(t, H));
    allocate(fused::group_bytes(t * K, e));
    allocate((size_t) t * K * sizeof(int32_t));
    allocate((size_t) (e + 1) * sizeof(int32_t));
    allocate((size_t) t * K * H * sizeof(float));
}
template<class Allocate> void device_layout(int64_t t, int e, Allocate allocate) {
    allocate(fused::act_bytes(t, H));
    allocate(fused::group_bytes(t * K, e));
    allocate((size_t) t * K * sizeof(int32_t));
    allocate(fused::act_bytes(t * K, FF));
    allocate((size_t) t * K * H * sizeof(float));
}
}

struct FusedPeer::Impl {
    int primary = -1, peer = -1, experts = 0;
    int64_t capacity = 0, returned = 0;
    void *host = nullptr, *device = nullptr;
    void *hx = nullptr, *hg = nullptr, *dx = nullptr, *dg = nullptr, *dh = nullptr;
    int32_t *hs = nullptr, *ho = nullptr, *ds = nullptr;
    float *hr = nullptr, *dr = nullptr;
    cudaStream_t compute = nullptr, land = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr, landed = nullptr;
    std::thread worker;
    std::string failure;
    ~Impl() {
        if (worker.joinable()) worker.join();
        if (primary >= 0) {
            const core::OnDevice on(primary);
            if (land) cudaStreamSynchronize(land);
            if (land) cudaStreamDestroy(land);
            if (ready) cudaEventDestroy(ready);
            if (landed) cudaEventDestroy(landed);
        }
        if (peer >= 0) {
            const core::OnDevice on(peer);
            if (compute) cudaStreamSynchronize(compute);
            if (compute) cudaStreamDestroy(compute);
            if (done) cudaEventDestroy(done);
            if (device) cudaFree(device);
        }
        if (host) cudaFreeHost(host);
    }
};

FusedPeer::FusedPeer() : impl_(new Impl) {}
FusedPeer::~FusedPeer() = default;
uint64_t FusedPeer::host_bytes(int64_t t, int e) {
    size_t bytes = 0;
    host_layout(t, e, [&](size_t n) { bytes += aligned(n); });
    return bytes;
}
uint64_t FusedPeer::device_bytes(int64_t t, int e) {
    size_t bytes = 0;
    device_layout(t, e, [&](size_t n) { bytes += aligned(n); });
    return bytes;
}
bool FusedPeer::open(int primary, int peer, int64_t tokens, int experts, std::string& error) {
    if (primary == peer || tokens <= 0 || experts < 1 || experts > 1024) {
        error = "fused peer: invalid geometry or devices"; return false;
    }
    impl_.reset(new Impl);
    auto& m = *impl_;
    m.primary = primary; m.peer = peer; m.capacity = tokens; m.experts = experts;
    try {
        const core::OnDevice on(primary);
        check(core::host_alloc(&m.host, host_bytes(tokens, experts), cudaHostAllocPortable));
        size_t at = 0; void* fields[5]; int f = 0;
        host_layout(tokens, experts, [&](size_t n) { fields[f++] = (uint8_t*) m.host + at; at += aligned(n); });
        m.hx = fields[0]; m.hg = fields[1]; m.hs = (int32_t*) fields[2];
        m.ho = (int32_t*) fields[3]; m.hr = (float*) fields[4];
        check(cudaStreamCreateWithFlags(&m.land, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&m.ready, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&m.landed, cudaEventDisableTiming));
        const core::OnDevice other(peer);
        check(cudaMalloc(&m.device, device_bytes(tokens, experts)));
        at = 0; f = 0;
        device_layout(tokens, experts, [&](size_t n) { fields[f++] = (uint8_t*) m.device + at; at += aligned(n); });
        m.dx = fields[0]; m.dg = fields[1]; m.ds = (int32_t*) fields[2];
        m.dh = fields[3]; m.dr = (float*) fields[4];
        check(cudaStreamCreateWithFlags(&m.compute, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&m.done, cudaEventDisableTiming));
        return true;
    } catch (const std::exception& e) { error = std::string("fused peer: ") + e.what(); return false; }
}

bool FusedPeer::start(std::vector<fused::Batch> batches, const fused::NativeGeom& g,
                      int64_t t, const void* x, const void* grouping, const int32_t* source,
                      float* output, void* primary_stream, std::string& error) {
    auto& m = *impl_;
    if (!m.device || t < 1 || t > m.capacity) { error = "fused peer: chunk exceeds capacity"; return false; }
    if (m.worker.joinable()) { error = "fused peer: preceding layer was not joined"; return false; }
    for (const auto& b : batches) if (b.e0 < 0 || b.e1 > m.experts || b.e1 <= b.e0 || b.e1 - b.e0 > fused::kMaxBatch) {
        error = "fused peer: invalid expert range"; return false;
    }
    try {
        const core::OnDevice on(m.primary);
        check(cudaStreamSynchronize(m.land)); // host buffers no longer feed the preceding layer
        const auto cs = (cudaStream_t) primary_stream;
        const size_t xb = fused::act_bytes(t, H), gb = fused::group_bytes(t * K, m.experts), sb = (size_t) t * K * 4;
        m.failure.clear(); m.returned = 0;
        check(cudaMemcpyAsync(m.hx, x, xb, cudaMemcpyDeviceToHost, cs));
        check(cudaMemcpyAsync(m.hg, grouping, gb, cudaMemcpyDeviceToHost, cs));
        check(cudaMemcpyAsync(m.hs, source, sb, cudaMemcpyDeviceToHost, cs));
        check(cudaMemcpyAsync(m.ho, fused::group_offsets(grouping, m.experts), (size_t) (m.experts + 1) * 4, cudaMemcpyDeviceToHost, cs));
        check(cudaEventRecord(m.ready, cs));
        m.worker = std::thread([&m, batches = std::move(batches), g, t, output, xb, gb, sb] {
            try {
                const core::OnDevice other(m.peer);
                check(cudaStreamWaitEvent(m.compute, m.ready, 0));
                check(cudaMemcpyAsync(m.dx, m.hx, xb, cudaMemcpyHostToDevice, m.compute));
                check(cudaMemcpyAsync(m.dg, m.hg, gb, cudaMemcpyHostToDevice, m.compute));
                check(cudaMemcpyAsync(m.ds, m.hs, sb, cudaMemcpyHostToDevice, m.compute));
                for (const auto& b : batches) {
                    fused::experts_native(b, g, m.experts, t * K, m.dg, m.dx, m.ds, m.dh, m.dr, m.compute);
                }
                check(cudaEventSynchronize(m.ready)); // offsets are now readable by the host
                for (const auto& b : batches) {
                    const size_t first = (size_t) m.ho[b.e0] * H, rows = (size_t) (m.ho[b.e1] - m.ho[b.e0]);
                    check(cudaMemcpyAsync(m.hr + first, m.dr + first, rows * H * 4, cudaMemcpyDeviceToHost, m.compute));
                    m.returned += (int64_t) rows;
                }
                check(cudaEventRecord(m.done, m.compute));
                const core::OnDevice on(m.primary);
                check(cudaStreamWaitEvent(m.land, m.done, 0));
                for (const auto& b : batches) {
                    const size_t first = (size_t) m.ho[b.e0] * H, rows = (size_t) (m.ho[b.e1] - m.ho[b.e0]);
                    check(cudaMemcpyAsync(output + first, m.hr + first, rows * H * 4, cudaMemcpyHostToDevice, m.land));
                }
                check(cudaEventRecord(m.landed, m.land));
            } catch (const std::exception& e) { m.failure = e.what(); }
        });
        return true;
    } catch (const std::exception& e) { error = std::string("fused peer: ") + e.what(); return false; }
}
bool FusedPeer::wait(void* primary_stream, std::string& error) {
    auto& m = *impl_;
    if (!m.worker.joinable()) { error = "fused peer: layer was not started"; return false; }
    m.worker.join();
    if (!m.failure.empty()) { error = "fused peer: " + m.failure; return false; }
    const core::OnDevice on(m.primary);
    const auto status = cudaStreamWaitEvent((cudaStream_t) primary_stream, m.landed, 0);
    if (status != cudaSuccess) { error = cudaGetErrorString(status); return false; }
    return true;
}
int64_t FusedPeer::rows() const { return impl_->returned; }

} // namespace basalt::prefill
