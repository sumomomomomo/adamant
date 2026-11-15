#include "state.hpp"
#include "layout.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <memory>

namespace basalt::prefill {
using namespace detail;
namespace { int large_peer_mode = 0; }
void Prefill::set_large_peer(int mode) { large_peer_mode = mode; }

void Prefill::set_ring_peer(core::PeerExperts* peer) {
    impl_->ring_peer = peer != nullptr && peer->valid() ? peer : nullptr;
}

void Prefill::set_ring_remote(const int32_t* res, const core::ExpertCache* cache, int device) {
    const bool on = res != nullptr && cache != nullptr && device >= 0;
    impl_->remote_res = on ? res : nullptr;
    impl_->remote_cache = on ? cache : nullptr;
    impl_->remote_dev = on ? device : -1;
}

void Prefill::set_peer_share(bool on) { impl_->peer_share = on; }
int64_t Prefill::peer_routed_rows_max() const { return impl_->pp ? impl_->pp->T_max : 0; }

void Prefill::set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
    if (layer_end < 0 && impl_->g != nullptr) layer_end = impl_->g->n_layers;   // after init: resolved here
    stage_lb_ = layer_begin;
    stage_le_ = layer_end;
    next_ = next;
    if (layer_begin == 0) {   // the first stage reads its tokens, not rows a stage before it left
        hand_in_ = nullptr;
        rows_in_place_ = false;
    }
}

// The peer card's prompt buffers for chunks of up to `Tp` tokens in groups of `G` rows: the shared transport's
// dimensions, in one place so that the tier's reserve and the allocation count the same bytes.
static PeerShare::Dims share_dims(int64_t Tp, int64_t G, const MmqPlan& mp) {
    PeerShare::Dims d;
    d.n_embd = N;
    d.n_ff = 640;
    d.group_bytes = peer_down_at(mp) + MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES;
    d.act_bytes = mmq::q8_bytes(Tp, N);
    d.max_tokens = (int) Tp;
    d.max_rows = Tp * K;   // no row cap: only the row tables grow with it
    d.n_expert = (int) NE;
    d.launch_experts = MMQ_GROUP;
    d.launch_rows = G;
    // one group buffer, as many as its own gathers had: the group's gather follows the last group's products
    d.stage_bufs = 1;
    d.bufs = 2;
    return d;
}
static bool make_peer(core::PeerExperts* peer, int64_t T_max, std::unique_ptr<PeerPrefill>& out,
                      std::string& err, bool optional = false) {
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->T_max = std::min<int64_t>(T_max, stream_all_min() - 1);
    pp->G = pp->T_max;   // one expert never has more rows than the chunk has tokens
    const int64_t Tp = pp->T_max;   // the longest chunk the peer takes
    pp->cap_rows = Tp * K;
    int prim = 0;
    cudaGetDevice(&prim);
    pp->dev = peer->device();
    const auto dims = share_dims(Tp, pp->G, mmq_plan());
    if (optional) {
        size_t main_free = 0, peer_free = 0, total = 0;
        cudaError_t status = cudaMemGetInfo(&main_free, &total);
        {
            const core::OnDevice od(pp->dev);
            if (status == cudaSuccess) status = cudaMemGetInfo(&peer_free, &total);
        }
        bool known = false;
        const uint64_t host_free = platform::ram_budget(&known);
        if (status != cudaSuccess || main_free < PeerShare::main_bytes(dims) + (64ull << 20) ||
            peer_free < PeerShare::peer_bytes(dims) + (128ull << 20) ||
            (known && host_free < PeerShare::host_bytes(dims) + (64ull << 20))) {
            (void) cudaGetLastError();
            err = "insufficient spare memory for the additional prefill share";
            return false;
        }
    }
    bool ok = true;
    size_t fb = 0, tb = 0;
    if (ok) {
        pp->share = std::make_unique<PeerShare>();
        ok = pp->share->open(prim, pp->dev, dims, err);
    }
    {
        const core::OnDevice od(pp->dev);
        cudaMemGetInfo(&fb, &tb);
    }
    if (!ok) {
        cudaGetLastError();
        if (err.empty()) err = "prefill peer: the peer's buffers do not fit";
        return false;
    }
    basalt::log::info("basalt prefill", "peer GPU %d computes its experts' rows of each prompt chunk up to %lld tokens, "
                         "with compact group buffers; %zu MiB left free on it", pp->dev, (long long) Tp, fb >> 20);
    out = std::move(pp);
    return true;
}

uint64_t Prefill::peer_bytes(int64_t chunk, bool streamed_share) {
    if (!mmq_plan().any) return 0;
    PeerPrefill pp;
    pp.T_max = std::min<int64_t>(chunk, streamed_share ? stream_all_min() - 1 : kPeerSlowRows);
    pp.G = pp.T_max;
    return PeerShare::peer_bytes(share_dims(pp.T_max, pp.G, mmq_plan())) +
           (large_peer_mode > 0 && chunk >= stream_all_min() ?
                FusedPeer::device_bytes(std::min(chunk, kFusedPeerChunkLimit)) : 0);
}

void Prefill::set_extra_peers(const std::vector<core::PeerExperts*>& peers, bool compute) {
    Impl& m = *impl_;
    m.extra_pp.clear();
    m.extra_peer_compute = false;
    for (auto* peer : peers) {
        if (peer == nullptr || !peer->valid()) continue;
        std::unique_ptr<PeerPrefill> pp;
        std::string err;
        if (!make_peer(peer, std::min(m.T_max, kExtraPeerChunkLimit), pp, err, true)) {
            basalt::log::warn("basalt prefill", "GPU %d: %s; additional peer disabled", peer->device(), err.c_str());
            continue;
        }
        m.extra_pp.push_back(std::move(pp));
    }
    m.extra_peer_compute = compute;
}

bool Prefill::set_peer(core::PeerExperts* peer, bool streamed_share, std::string& err) {
    Impl& m = *impl_;
    if (peer == nullptr || !peer->valid()) { m.pp.reset(); return true; }
    // The peer takes the rows of the routed-only chunks (below stream_all_min()): the activations and
    // result rows cross through the share's host staging and the streamed ring stays off.
    if (!mmq_plan().any) { err = "prefill peer: needs the MMQ prompt path"; return false; }
    std::unique_ptr<PeerPrefill> pp;
    const int64_t t_routed = streamed_share ? m.T_max : std::min<int64_t>(m.T_max, kPeerSlowRows);
    bool ok = make_peer(peer, t_routed, pp, err);
    if (!ok) return false;
    m.pp = std::move(pp);
    if (large_peer_mode > 0 && m.T_max >= stream_all_min()) {
        const int64_t capacity = std::min(m.T_max, kFusedPeerChunkLimit);
        m.large_peer_compute = large_peer_mode == 1;
        m.large_peer = std::make_unique<FusedPeer>();
        if (!m.large_peer->open(m.device, peer->device(), capacity, (int) m.g->n_expert, err)) {
            m.large_peer.reset(); m.large_peer_compute = false; return false;
        }
        basalt::log::info("basalt prefill", "large peer GPU %d: up to %lld tokens, %llu MiB device, %llu MiB host",
                          peer->device(), (long long) capacity,
                          (unsigned long long) (FusedPeer::device_bytes(capacity) >> 20),
                          (unsigned long long) (FusedPeer::host_bytes(capacity) >> 20));
    }
    return true;
}

}  // namespace basalt::prefill
