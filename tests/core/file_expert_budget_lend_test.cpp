#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
struct Temp {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("basalt-budget-lend-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temp() { std::filesystem::create_directories(path); }
    ~Temp() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

void run(bool partial, bool spares, bool pin, bool budgeted) {
    namespace cpu = basalt::kernels::cpu;
    using namespace basalt::core;
    Temp tmp;
    std::string error;
    cpu::NativeFmt f[2];
    check(cpu::native_fmt(GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_0, 2560, 640, f[0], error), error);
    check(cpu::native_fmt(GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, 2560, 640, f[1], error), error);
    check(f[0].bytes != f[1].bytes, "fixture must exercise unequal expert sizes");
    std::ostringstream layout;
    layout << "0 " << GGML_TYPE_IQ4_XS << ' ' << GGML_TYPE_Q5_0 << " 0 " << f[0].bytes << '\n';
    layout << "1 " << GGML_TYPE_IQ2_S << ' ' << GGML_TYPE_Q2_0 << ' ' << 4 * f[0].bytes << ' ' << f[1].bytes << '\n';
    check(cpu::expert_layout_load_text(layout.str(), 2, 4, error, 2560, 640), error);
    {
        std::ofstream out(tmp.path / "experts.bin", std::ios::binary);
        for (int l = 0; l < 2; ++l) for (int e = 0; e < 4; ++e) {
            std::vector<char> blob(f[l].bytes, (char) (l * 4 + e + 1));
            out.write(blob.data(), blob.size());
        }
        check((bool) out, "write fixture");
    }
    FileExpertSource src;
    check(src.open(FileRange((tmp.path / "experts.bin").string()), 2, 4, error), error);
    ExpertCache cache;
    check(cache.open_sized({(int64_t) f[0].bytes, (int64_t) f[0].bytes,
                            (int64_t) f[1].bytes, (int64_t) f[1].bytes}, 2, 4, error), error);
    std::vector<std::pair<int32_t, int32_t>> rank;
    for (int l = 0; l < 2; ++l) for (int e = 0; e < 4; ++e) {
        rank.emplace_back(l, e);
        if (e < 2) {
            const int slot = cache.admit(l, e);
            check(slot == l * 2 + e, "fixture slot order");
            check(cache.fill_slot_blocking(slot, src.blob(l, e), error, f[l].bytes), error);
        }
    }
    // Duplicate and invalid ranks cannot duplicate copies or consume extra budget.
    rank.emplace_back(0, 2); rank.emplace_back(-1, 0); rank.emplace_back(2, 0);
    const uint64_t tail = f[0].bytes + 2 * f[1].bytes;
    const uint64_t spare = spares ? 8 * (f[0].bytes + f[1].bytes) : 0;
    if (spares) src.set_spares(2);
    const uint64_t cap = partial ? f[1].bytes + std::min(f[0].bytes, f[1].bytes) - 1 : tail + f[0].bytes + spare;
    check(src.pin_cache_complement(cache, error, pin, {}, 1, 0, budgeted ? cap : 0, &rank), error);
    check(src.resident_lent_slots() == (partial ? 1 : 3), "wrong contiguous backed tail");
    check(!budgeted || src.resident_bytes() <= cap, "copies and spare places exceeded the requested budget");
    if (partial) {
        check(src.has_resident(1, 1) && !src.has_resident(1, 0) && !src.has_resident(0, 1), "partial tail has a hole");
    } else {
        check(!src.has_resident(0, 0), "unlent primary expert unnecessarily copied");
        check(src.has_resident(0, 1) && src.has_resident(1, 0) && src.has_resident(1, 1), "tail is not fully backed");
        check(src.has_resident(0, 2), "remaining budget did not fund the hottest missing expert");
        if (budgeted) check(!src.has_resident(0, 3) && !src.has_resident(1, 2), "hot selection exceeded its share");
        else check(src.has_resident(0, 3) && src.has_resident(1, 2) && src.has_resident(1, 3), "whole-complement behavior changed");
    }
    if (spares) {
        uint64_t offset = 0;
        check(src.spare_take(0, offset) != nullptr, "adaptive places were not reserved");
        src.spare_return(0, offset);
    }
    // Borrowing destroys VRAM contents; a restore must use RAM and reproduce the original expert bytes.
    for (int slot = partial ? 3 : 1; slot < 4; ++slot) {
        const int l = slot / 2, e = slot % 2;
        int fd = -1; uint64_t offset = 0;
        check(!src.file_extent(l, e, fd, offset), "backed expert still uses file I/O");
        check(cudaMemset((void*) cache.device_slot(slot), 0xa5, f[l].bytes) == cudaSuccess, "overwrite loan");
        check(cache.fill_slot_blocking(slot, src.blob(l, e), error, f[l].bytes), error);
        std::vector<uint8_t> actual(f[l].bytes);
        check(cudaMemcpy(actual.data(), cache.device_slot(slot), actual.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "read restored slot");
        check(std::all_of(actual.begin(), actual.end(), [&](uint8_t v) { return v == l * 4 + e + 1; }), "restored weights differ");
    }
    std::printf("partial=%d spares=%d pin=%d budgeted=%d: PASS\n", partial, spares, pin, budgeted);
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        run(false, false, false, true);
        run(true, false, false, true);
        run(false, true, false, true);
        run(false, false, true, true);
        run(false, false, false, false);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
