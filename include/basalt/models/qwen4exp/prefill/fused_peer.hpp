#pragma once

#include "basalt/models/qwen4exp/prefill/moe_fused_iq.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::prefill {

// Copies the primary's quantized inputs and row map, computes disjoint expert
// ranges, and returns their unchanged rows for the primary's ordinary combine.
class FusedPeer {
public:
    FusedPeer();
    ~FusedPeer();
    FusedPeer(const FusedPeer&) = delete;
    FusedPeer& operator=(const FusedPeer&) = delete;
    static uint64_t device_bytes(int64_t tokens, int experts = 512);
    static uint64_t host_bytes(int64_t tokens, int experts = 512);
    bool open(int primary, int peer, int64_t tokens, int experts, std::string& error);
    bool start(std::vector<fused::Batch> batches, const fused::NativeGeom& geometry,
               int64_t tokens, const void* quantized, const void* grouping, const int32_t* source,
               float* output, void* primary_stream, std::string& error);
    bool wait(void* primary_stream, std::string& error);
    int64_t rows() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace basalt::prefill
