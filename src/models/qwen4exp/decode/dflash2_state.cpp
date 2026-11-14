#include "basalt/models/qwen4exp/decode/dflash2.hpp"
#include "basalt/core/on_device.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <new>

namespace basalt::core {
namespace {
using Header = std::array<uint64_t, 8>;
constexpr uint64_t kMagic = 0x31545332464c4644ull;
constexpr uint64_t kVersion = 1;
bool status(cudaError_t code, std::string& error) {
    if (code == cudaSuccess) return true;
    error = std::string("dflash2 snapshot: ") + cudaGetErrorString(code);
    return false;
}
}

size_t Dflash2Drafter::state_bytes(int64_t position) const {
    if (position < 0) return 0;
    const size_t rows = (size_t) std::min<int64_t>(position, window_ - 1);
    return sizeof(Header) + (size_t) L_ * 2 * NKV_ * rows * HD_ * sizeof(float);
}

bool Dflash2Drafter::validate_state(const CheckpointBytes& bytes, int64_t position, std::string& error) const {
    Header h{};
    if (position < 0 || bytes.size() < sizeof h) {
        error = "dflash2 snapshot: missing or truncated header";
        return false;
    }
    std::memcpy(h.data(), bytes.data(), sizeof h);
    const uint64_t rows = (uint64_t) std::min<int64_t>(position, window_ - 1);
    const Header expected{kMagic, kVersion, artifact_id_, (uint64_t) position, rows,
                          (uint64_t) L_, (uint64_t) (NKV_ * HD_), (uint64_t) window_};
    if (h != expected || bytes.size() != state_bytes(position)) {
        error = "dflash2 snapshot: incompatible identity, position, geometry or payload size";
        return false;
    }
    return true;
}

bool Dflash2Drafter::transfer_state(uint8_t* host, int64_t position, bool restore, std::string& error) {
    const size_t rows = (size_t) std::min<int64_t>(position, window_ - 1);
    if (rows == 0) return true;
    const size_t unit = (size_t) HD_ * sizeof(float);
    const size_t first = (size_t) ((position - (int64_t) rows) % kRing);
    const size_t tail = std::min(rows, (size_t) kRing - first);
    for (int l = 0; l < L_; ++l) {
        for (float* ring : {kring_[l], vring_[l]}) {
            auto* device = reinterpret_cast<uint8_t*>(ring);
            auto copy = [&](size_t slot, size_t at, size_t count) {
                if (count == 0) return true;
                return restore
                    ? status(cudaMemcpy2DAsync(device + slot * unit, kRing * unit, host + at * unit,
                                                rows * unit, count * unit, NKV_, cudaMemcpyHostToDevice, cs_), error)
                    : status(cudaMemcpy2DAsync(host + at * unit, rows * unit, device + slot * unit,
                                                kRing * unit, count * unit, NKV_, cudaMemcpyDeviceToHost, cs_), error);
            };
            if (!copy(first, 0, tail) || !copy(0, tail, rows - tail)) return false;
            host += (size_t) NKV_ * rows * unit;
        }
    }
    return status(cudaStreamSynchronize(cs_), error);
}

bool Dflash2Drafter::save_state(CheckpointBytes& bytes, int64_t position, std::string& error) {
    if (position < 0 || position != have_ctx_ || !bound_) {
        error = "dflash2 snapshot: context is not at the target checkpoint";
        return false;
    }
    const OnDevice on(device_);
    if (!status(cudaStreamSynchronize(cs_), error)) return false;
    try { bytes.resize(state_bytes(position)); }
    catch (const std::bad_alloc&) { error = "dflash2 snapshot: host allocation failed"; return false; }
    const Header h{kMagic, kVersion, artifact_id_, (uint64_t) position,
                   (uint64_t) std::min<int64_t>(position, window_ - 1),
                   (uint64_t) L_, (uint64_t) (NKV_ * HD_), (uint64_t) window_};
    std::memcpy(bytes.data(), h.data(), sizeof h);
    return transfer_state(bytes.data() + sizeof h, position, false, error);
}

bool Dflash2Drafter::restore_state(const CheckpointBytes& bytes, int64_t position, std::string& error) {
    if (!validate_state(bytes, position, error)) return false;
    const OnDevice on(device_);
    if (!status(cudaStreamSynchronize(cs_), error) ||
        !transfer_state(const_cast<uint8_t*>(bytes.data()) + sizeof(Header), position, true, error)) return false;
    have_ctx_ = position;
    return true;
}
}
