#include "basalt/models/qwen4exp/prefill/stage_weights.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace basalt::core {
struct WeightTableFixture {
    static void put(WeightTable& table, const std::string& name, const WeightRef& ref) { table.table_[name] = ref; }
};
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
    std::mt19937 random(71);
    using namespace basalt;
    for (const auto& [rows, cols] : {std::pair{320, 10240}, std::pair{10240, 320}}) {
        const size_t n = (size_t) rows * cols;
        const int scale_stride = ((cols / 32) + 7) & ~7;
        std::vector<int8_t> q(n);
        std::vector<uint16_t> scales((size_t) rows * scale_stride, 0);
        std::vector<uint16_t> want(n), got(n);
        for (int r = 0; r < rows; ++r)
            for (int group = 0; group < cols / 32; ++group)
                scales[(size_t) r * scale_stride + group] = random() % 8 == 0 ? 0 :
                    (uint16_t) (0x1000 + random() % 0x3000);
        for (size_t i = 0; i < n; ++i) {
            q[i] = (int8_t) ((int) (random() % 255) - 127);
            const uint16_t h = scales[(i / cols) * scale_stride + (i % cols) / 32];
            const float scale = h == 0 ? 0.0f : std::ldexp((float) (1024 + (h & 1023)), (h >> 10) - 25);
            const float x = (float) q[i] * scale;
            uint32_t bits;
            std::memcpy(&bits, &x, sizeof bits);
            want[i] = (uint16_t) ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        }
        cudaSetDevice(0);
        const void* source = kernels::hc8_upload(nullptr, q.data(), scales.data(), rows, cols);
        if (source == nullptr) return 1;
        cudaSetDevice(1);
        uint16_t* destination = nullptr;
        if (cudaMalloc((void**) &destination, n * sizeof(uint16_t)) != cudaSuccess) return 1;
        cudaMemset(destination, 0xff, n * sizeof(uint16_t));
        core::WeightTable main, stage;
        core::WeightRef a, b;
        a.kind = b.kind = core::WeightKind::Bf16InF32;
        a.ne0 = b.ne0 = cols;
        a.ne1 = b.ne1 = rows;
        a.data = source;
        b.data = destination;
        core::WeightTableFixture::put(main, "blk.26.hc_attn_down.weight", a);
        core::WeightTableFixture::put(stage, "blk.26.hc_attn_down.weight", b);
        std::string error;
        if (!models::qwen4exp::mirror_prompt_hc8(main, stage, 1, error) ||
            cudaMemcpy(got.data(), destination, n * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
            got != want) {
            std::fprintf(stderr, "prompt stage weights: %s\n", error.c_str());
            return 1;
        }
        b.ne0 = cols + 1;
        core::WeightTableFixture::put(stage, "blk.26.hc_attn_down.weight", b);
        if (models::qwen4exp::mirror_prompt_hc8(main, stage, 1, error)) return 1;
        cudaFree(destination);
    }
    std::puts("prompt stage weights: HC8 values byte-identical to CPU reference on peer, both matrix shapes");
    return 0;
}
