#pragma once

#include <cstdint>
#include <sstream>
#include <string>

namespace basalt::prefill {

struct Cutoffs {
    int64_t stream_min = 0;
    int64_t extra_max = 2048;
    int64_t large_max = 4096;
    bool reserve_wide = false;

    int64_t routed_capacity(int64_t stream) const { return reserve_wide ? 8191 : stream - 1; }
    int64_t extra_capacity() const { return reserve_wide ? 6144 : extra_max; }
    int64_t large_capacity() const { return reserve_wide ? 8192 : large_max; }
    bool reserve_large(int64_t stream) const { return stream_min == 0 || large_capacity() >= stream; }
};

inline Cutoffs serving_cutoffs(bool native_next, bool extra_compute, int peers, bool streamed_share, bool exclusive) {
    Cutoffs c;
    if (native_next && extra_compute && peers == 2 && streamed_share && !exclusive) {
        c.stream_min = 6144;
        c.extra_max = 6143;
    }
    return c;
}

inline int prompt_stage_limit(int peers, bool all_alike, bool native_next, bool first_alike) {
    return peers <= 1 ? 2 : all_alike ? peers + 1 :
           native_next && peers == 2 && first_alike ? 2 : 1;
}

inline int paired_stage_layer_floor(bool native_next, int peers, int stage_limit, bool first_alike,
                                    int layers, int width) {
    return native_next && peers == 2 && stage_limit == 2 && first_alike && layers == 48 && width == 2560 ? 26 : 0;
}

inline int64_t paired_stage_min_tokens(int64_t chunk_cap) {
    return (chunk_cap == 32768 ? chunk_cap : 2 * chunk_cap) + 1;
}

inline Cutoffs& cutoffs() { static Cutoffs value; return value; }

inline bool parse_cutoffs(const std::string& text, Cutoffs& out) {
    Cutoffs c;
    char comma1 = 0, comma2 = 0;
    std::istringstream in(text);
    if (!(in >> c.stream_min >> comma1 >> c.extra_max >> comma2 >> c.large_max) ||
        comma1 != ',' || comma2 != ',' || !(in >> std::ws).eof() ||
        c.stream_min < 128 || c.stream_min > 8192 || c.extra_max < 128 || c.extra_max > 6144 ||
        c.extra_max >= c.stream_min || c.large_max < 128 || c.large_max > 8192) return false;
    c.reserve_wide = true;
    out = c;
    return true;
}

}  // namespace basalt::prefill
