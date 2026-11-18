#include "basalt/models/qwen4exp/prefill/policy.hpp"
#include <cstdio>

int main() {
    using basalt::prefill::Cutoffs;
    using basalt::prefill::parse_cutoffs;
    int failed = 0;
    auto check = [&](bool ok) { if (!ok) ++failed; };
    for (int peers : {0, 1, 2, 3}) for (bool alike : {false, true})
        for (bool next : {false, true}) for (bool first : {false, true}) {
            const int cap = basalt::prefill::prompt_stage_limit(peers, alike, next, first);
            check(cap == (peers <= 1 ? 2 : alike ? peers + 1 : next && peers == 2 && first ? 2 : 1));
            for (int layers : {32, 48}) for (int width : {2048, 2560}) {
                const int floor = basalt::prefill::paired_stage_layer_floor(next, peers, cap, first, layers, width);
                check(floor == (next && peers == 2 && !alike && first && layers == 48 && width == 2560 ? 26 : 0));
            }
        }
    Cutoffs c;
    check(c.extra_max == 2048 && c.large_max == 4096 && c.reserve_large(6144));
    check(c.routed_capacity(2560) == 2559 && c.extra_capacity() == 2048 && c.large_capacity() == 4096);
    check(parse_cutoffs("6144,6143,8192", c));
    check(c.stream_min == 6144 && c.extra_max == 6143 && c.large_max == 8192);
    check(c.routed_capacity(2560) == 8191 && c.extra_capacity() == 6144 && c.large_capacity() == 8192);
    for (const char* text : {"", "6144,6144,8192", "127,128,4096", "8193,2048,4096", "6144,6145,4096",
                             "6144,2048,8193", "6144,2048,127", "6144:2048:4096", "6144,2048,4096junk",
                             "6144,2048,4096,1", "6144,2048", "6144,-1,4096"}) {
        Cutoffs old = c;
        check(!parse_cutoffs(text, c));
        check(c.stream_min == old.stream_min && c.extra_max == old.extra_max && c.large_max == old.large_max);
    }
    for (const char* text : {"2560,2048,4096", "4096,4095,4096", "8192,6144,8192"}) {
        check(parse_cutoffs(text, c));
        check(c.routed_capacity(c.stream_min) >= c.stream_min - 1);
        check(c.extra_capacity() >= c.extra_max && c.large_capacity() >= c.large_max);
    }
    for (bool next : {false, true}) for (bool extra : {false, true})
        for (int peers : {0, 1, 2, 3}) for (bool share : {false, true}) for (bool exclusive : {false, true}) {
            const auto p = basalt::prefill::serving_cutoffs(next, extra, peers, share, exclusive);
            const bool tuned = next && extra && peers == 2 && share && !exclusive;
            check(p.stream_min == (tuned ? 6144 : 0));
            check(p.extra_max == (tuned ? 6143 : 2048) && p.large_max == 4096);
            check(!p.reserve_wide);
            if (tuned) {
                check(p.routed_capacity(p.stream_min) >= p.extra_capacity());
                check(!p.reserve_large(p.stream_min) && p.reserve_large(128));
            } else check(p.reserve_large(6144));
        }
    std::printf("prefill_policy_test: %d failures\n", failed);
    return failed ? 1 : 0;
}
