#pragma once

#include "basalt/core/on_device.hpp"
#include "basalt/core/host_plan.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::qwen4exp {

// Collects the columns of prompt features produced by separate layer stages.
class PromptTapRelay {
public:
    ~PromptTapRelay() { clear(); }
    PromptTapRelay() = default;
    PromptTapRelay(const PromptTapRelay&) = delete;
    PromptTapRelay& operator=(const PromptTapRelay&) = delete;

    bool init(const std::vector<int>& devices, std::vector<int> layers, int width, int capacity,
              std::string& error) {
        clear();
        layers_ = std::move(layers);
        width_ = width;
        capacity_ = capacity;
        if (devices.empty() || layers_.empty() || width <= 0 || capacity <= 0) {
            error = "prompt features: invalid relay geometry";
            return false;
        }
        for (int device : devices) {
            const core::OnDevice on(device);
            buffers_.push_back({device});
            Buffer& b = buffers_.back();
            if (!check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking), error)) return false;
            if (buffers_.size() == 1) {
                if (!check(core::host_alloc((void**) &b.host, bytes(capacity), cudaHostAllocMapped | cudaHostAllocPortable), error) ||
                    !check(cudaHostGetDevicePointer((void**) &b.data, b.host, 0), error)) return false;
            } else if (!check(cudaMalloc((void**) &b.data, bytes(capacity)), error)) return false;
        }
        return true;
    }

    float* data(int device) const {
        for (const Buffer& b : buffers_) if (b.device == device) return b.data;
        return nullptr;
    }
    size_t elements(int rows) const { return (size_t) rows * layers_.size() * (size_t) width_; }

    // The producer's stream has completed these rows. Other stages own different columns.
    bool read(int device, const float* source, int lo, int hi, int rows, std::vector<float>& host,
              std::string& error) {
        Buffer* b = buffer(device);
        if (b == nullptr || source == nullptr || rows < 0 || rows > capacity_ || host.size() < elements(rows)) {
            error = "prompt features: invalid relay read";
            return false;
        }
        const core::OnDevice on(device);
        const size_t pitch = layers_.size() * (size_t) width_ * sizeof(float);
        for (size_t i = 0; i < layers_.size(); ++i)
            if (layers_[i] >= lo && layers_[i] < hi &&
                !check(cudaMemcpy2DAsync(host.data() + i * width_, pitch, source + i * width_, pitch,
                                        (size_t) width_ * sizeof(float), rows, cudaMemcpyDefault, b->stream), error))
                return false;
        return check(cudaStreamSynchronize(b->stream), error);
    }

    bool upload(int device, const std::vector<float>& host, int rows, std::string& error) {
        Buffer* b = buffer(device);
        if (b == nullptr || rows < 0 || rows > capacity_ || host.size() < elements(rows)) {
            error = "prompt features: invalid relay upload";
            return false;
        }
        if (b->host != nullptr) {
            std::memcpy(b->host, host.data(), bytes(rows));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            return true;
        }
        const core::OnDevice on(device);
        return check(cudaMemcpyAsync(b->data, host.data(), bytes(rows), cudaMemcpyHostToDevice, b->stream), error) &&
               check(cudaStreamSynchronize(b->stream), error);
    }

private:
    struct Buffer { int device = -1; float* data = nullptr; float* host = nullptr; cudaStream_t stream = nullptr; };
    std::vector<Buffer> buffers_;
    std::vector<int> layers_;
    int width_ = 0, capacity_ = 0;
    size_t bytes(int rows) const { return elements(rows) * sizeof(float); }
    Buffer* buffer(int device) {
        for (Buffer& b : buffers_) if (b.device == device) return &b;
        return nullptr;
    }
    static bool check(cudaError_t result, std::string& error) {
        if (result == cudaSuccess) return true;
        error = std::string("prompt features: ") + cudaGetErrorString(result);
        return false;
    }
    void clear() {
        for (Buffer& b : buffers_) {
            const core::OnDevice on(b.device);
            if (b.stream != nullptr) { cudaStreamSynchronize(b.stream); cudaStreamDestroy(b.stream); }
            if (b.host != nullptr) cudaFreeHost(b.host);
            else if (b.data != nullptr) cudaFree(b.data);
        }
        buffers_.clear();
    }
};

}  // namespace basalt::models::qwen4exp
