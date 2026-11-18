#include "basalt/models/qwen4exp/prefill/tap_relay.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void copy_features(const float* source, float* output, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = source[i];
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    constexpr int width = 2560, capacity = 2048;
    const std::vector<int> layers{8, 17, 26, 35, 44};
    basalt::models::qwen4exp::PromptTapRelay relay;
    std::string error;
    if (!relay.init({0, 1}, layers, width, capacity, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    float* output = nullptr;
    cudaSetDevice(0);
    if (cudaMalloc((void**) &output, relay.elements(capacity) * sizeof(float)) != cudaSuccess) return 1;
    for (int rows : {1, 33, capacity}) {
        const size_t n = relay.elements(rows);
        std::vector<float> a(n), b(n), want(n), got(n), joined(n, -1.0f);
        for (size_t i = 0; i < n; ++i) {
            a[i] = (float) (i % 104729) * 0.0001f;
            b[i] = -a[i] - 0.125f;
        }
        for (size_t i = 0; i < n; ++i) want[i] = layers[i / width % layers.size()] < 23 ? a[i] : b[i];
        for (int device : {0, 1}) {
            const basalt::core::OnDevice on(device);
            const auto& source = device == 0 ? a : b;
            if (cudaMemcpy(relay.data(device), source.data(), n * sizeof(float), cudaMemcpyDefault) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess)
                return 1;
        }
        if (!relay.read(0, relay.data(0), 0, 23, rows, joined, error) ||
            !relay.read(1, relay.data(1), 23, 48, rows, joined, error) ||
            !relay.upload(0, joined, rows, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        const basalt::core::OnDevice on0(0);
        copy_features<<<(unsigned)((n + 255) / 256), 256>>>(relay.data(0), output, n);
        if (cudaMemcpy(got.data(), output, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            std::memcmp(want.data(), got.data(), n * sizeof(float)) != 0) {
            std::fprintf(stderr, "prompt tap relay: mismatched columns for %d rows\n", rows);
            return 1;
        }
    }
    cudaFree(output);
    std::vector<float> short_host(1);
    if (relay.read(0, relay.data(0), 0, 48, capacity, short_host, error) ||
        relay.upload(0, short_host, capacity + 1, error)) return 1;
    std::puts("prompt tap relay: byte-identical split columns at real geometry");
    return 0;
}
