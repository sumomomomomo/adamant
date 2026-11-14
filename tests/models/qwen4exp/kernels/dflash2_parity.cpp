#include "basalt/models/qwen4exp/decode/layer.hpp"
#include "basalt/models/qwen4exp/kernels/dflash2.hpp"
#include "basalt/models/qwen4exp/kernels/gr.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using namespace basalt::kernels;
namespace {
void cu(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Dev {
    T *p{};
    size_t n;
    explicit Dev(size_t n) : n(n) {
        cu(cudaMalloc((void **)&p, n * sizeof(T)));
        cu(cudaMemset(p, 0, n * sizeof(T)));
    }
    explicit Dev(const std::vector<T> &v) : Dev(v.size()) { put(v); }
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T> &v) { cu(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        cu(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
std::mt19937 rng(17041);
std::vector<float> randoms(size_t n, float scale = 1) {
    std::normal_distribution<float> d(0, scale);
    std::vector<float> v(n);
    for (auto &x : v)
        x = d(rng);
    return v;
}
void compare(const char *name, const std::vector<float> &got, const std::vector<double> &ref, double atol = 2e-5,
             double rtol = 2e-5) {
    double ma = 0, mr = 0;
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = std::abs(got[i] - ref[i]);
        ma = std::max(ma, a);
        mr = std::max(mr, a / std::max(1e-8, std::abs(ref[i])));
        bad += !std::isfinite(got[i]) || a > atol + rtol * std::abs(ref[i]);
    }
    std::printf("%s max_abs %.8g max_rel %.8g bad %zu\n", name, ma, mr, bad);
    if (bad)
        throw std::runtime_error(name);
}
void bench(const char *name, size_t bytes, const std::function<void()> &fn) {
    fn();
    cu(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cu(cudaEventCreate(&a));
    cu(cudaEventCreate(&b));
    cu(cudaEventRecord(a));
    for (int i = 0; i < 30; ++i)
        fn();
    cu(cudaEventRecord(b));
    cu(cudaEventSynchronize(b));
    float ms;
    cu(cudaEventElapsedTime(&ms, a, b));
    double us = ms * 1000 / 30;
    std::printf("%s %.3f us %.3f GB/s\n", name, us, bytes / us / 1000);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}
std::vector<float> dq(const Dev<int8_t> &q, const Dev<float> &d, int K, int rows) {
    auto a = q.get();
    auto s = d.get();
    std::vector<float> v(rows * K);
    for (int i = 0; i < rows * K; ++i)
        v[i] = a[i] * s[i / 32];
    return v;
}
void quant_check(const char *name, const Dev<int8_t> &q, const Dev<float> &d, const std::vector<double> &ref, int K,
                 int rows) {
    auto got = dq(q, d, K, rows);
    double ma = 0, mr = 0;
    int bad = 0;
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < K / 32; ++b) {
            double mx = 0;
            for (int j = 0; j < 32; ++j)
                mx = std::max(mx, std::abs(ref[r * K + b * 32 + j]));
            for (int j = 0; j < 32; ++j) {
                int i = r * K + b * 32 + j;
                double e = std::abs(got[i] - ref[i]);
                ma = std::max(ma, e);
                mr = std::max(mr, e / std::max(1e-8, std::abs(ref[i])));
                bad += e > mx / 254 + 3e-5;
            }
        }
    std::printf("%s max_abs %.8g max_rel %.8g bad %d\n", name, ma, mr, bad);
    if (bad)
        throw std::runtime_error(name);
}
std::vector<double> norm(const std::vector<double> &x, const std::vector<float> &g, int N, int rows) {
    auto y = x;
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int i = 0; i < N; ++i)
            ss += x[r * N + i] * x[r * N + i];
        double s = 1 / std::sqrt(ss / N + 1e-6);
        for (int i = 0; i < N; ++i)
            y[r * N + i] = x[r * N + i] * s * g[i];
    }
    return y;
}
void elementwise() {
    const int H = 2560, B = 8, G = H / 16;
    auto x = randoms(B * H), h = randoms(B * H), dy = randoms(B * 4 * G, .1f), base = randoms(4 * H, .1f),
         gamma = randoms(H, .3f);
    Dev<float> dx(x), dh(h), dd(dy), db(base), dg(gamma), out(B * H), xn(B * H), sc(B * H / 32);
    Dev<int8_t> q(B * H);
    std::vector<double> conv(B * H), res(B * H);
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < H; ++c) {
            double a = 0;
            for (int o = 0; o < 2 && o <= r; ++o)
                a += (double(base[o * H + c]) + dy[r * 4 * G + o * G + c / 16]) * x[(r - o) * H + c];
            conv[r * H + c] = a;
            res[r * H + c] = a + h[r * H + c];
        }
    df2_conv(dx.p, dd.p, 4 * G, db.p, dh.p, out.p, B, H, 16, nullptr);
    compare("conv", out.get(), res);
    df2_conv_quant(dx.p, dd.p, 4 * G, db.p, B, H, 16, q.p, sc.p, nullptr);
    quant_check("conv_quant", q, sc, conv, H, B);
    auto rn = norm(res, gamma, H, B);
    df2_conv_rms_quant(dx.p, dd.p, 4 * G, db.p, dh.p, dg.p, xn.p, B, H, 16, 1e-6f, q.p, sc.p, nullptr);
    compare("conv_rms residual", dh.get(), res);
    compare("conv_rms norm", xn.get(), rn);
    quant_check("conv_rms quant", q, sc, rn, H, B);
    df2_rms_quant(dx.p, dg.p, xn.p, B, H, 1e-6f, q.p, sc.p, nullptr);
    auto nx = norm(std::vector<double>(x.begin(), x.end()), gamma, H, B);
    compare("rms", xn.get(), nx);
    quant_check("rms_quant", q, sc, nx, H, B);
    df2_quant_rows(dx.p, H, B, q.p, sc.p, nullptr);
    quant_check("quant_rows", q, sc, std::vector<double>(x.begin(), x.end()), H, B);
    bench("rms_quant", B * H * 9, [&] { df2_rms_quant(dx.p, dg.p, xn.p, B, H, 1e-6f, q.p, sc.p, nullptr); });
    bench("conv_quant", B * H * 9, [&] { df2_conv_quant(dx.p, dd.p, 4 * G, db.p, B, H, 16, q.p, sc.p, nullptr); });
    bench("conv_rms_quant", B * H * 17,
          [&] { df2_conv_rms_quant(dx.p, dd.p, 4 * G, db.p, dh.p, dg.p, xn.p, B, H, 16, 1e-6f, q.p, sc.p, nullptr); });
    const int I = 7680;
    auto gate = randoms(B * I), up = randoms(B * I);
    Dev<float> ga(gate), u(up), sd(B * I / 32);
    Dev<int8_t> qi(B * I);
    std::vector<double> sw(B * I);
    for (int i = 0; i < B * I; ++i)
        sw[i] = gate[i] / (1 + std::exp(-double(gate[i]))) * up[i];
    df2_swiglu_quant(ga.p, u.p, B, I, qi.p, sd.p, nullptr);
    quant_check("swiglu_quant", qi, sd, sw, I, B);
    bench("swiglu_quant", B * I * 9, [&] { df2_swiglu_quant(ga.p, u.p, B, I, qi.p, sd.p, nullptr); });
}
void taps() {
    const int N = 2560, HC = 4, S = 5 * N;
    for (int B : {1, 2, 7, 8}) {
        auto r = randoms(B * HC * N), bo = randoms(B * N), inj = randoms(B * HC);
        Dev<float> dr(r), db(bo), di(inj), out(B * S);
        std::vector<double> ref(B * S, 0);
        for (int t = 0; t < B; ++t)
            for (int i = 0; i < N; ++i) {
                double a = 0;
                for (int c = 0; c < HC; ++c)
                    a += r[(t * HC + c) * N + i];
                ref[t * S + i] = a / HC;
            }
        df2_tap_rows(dr.p, out.p, S, N, HC, B, nullptr);
        compare("tap_rows", out.get(), ref);
        for (int t = 0; t < B; ++t)
            for (int i = 0; i < N; ++i) {
                double a = 0;
                for (int c = 0; c < HC; ++c)
                    a += r[(t * HC + c) * N + i] + bo[t * N + i] * 2 / (1 + std::exp(-double(inj[t * HC + c]) / HC));
                ref[t * S + i] = a / HC;
            }
        df2_tap_gr(dr.p, HC * N, db.p, N, di.p, HC, out.p, S, B, N, HC, nullptr);
        compare("tap_gr", out.get(), ref);
        // The production gated write, then an ordinary tap, is the existing-kernel
        // oracle.
        basalt::kernels::GrShapes gs;
        gs.n_embd = N;
        gs.hc = HC;
        for (int t = 0; t < B; ++t)
            gr_write(dr.p + t * HC * N, db.p + t * N, di.p + t * HC, gs, dr.p + t * HC * N, nullptr);
        Dev<float> prev(B * S);
        df2_tap_rows(dr.p, prev.p, S, N, HC, B, nullptr);
        auto a = prev.get();
        compare("tap_gr vs gr_write", out.get(), std::vector<double>(a.begin(), a.end()));
        bench("tap_rows", B * HC * N * 4, [&] { df2_tap_rows(dr.p, out.p, S, N, HC, B, nullptr); });
        bench("tap_gr", B * (HC + 1) * N * 4,
              [&] { df2_tap_gr(dr.p, HC * N, db.p, N, di.p, HC, out.p, S, B, N, HC, nullptr); });
    }
}
void rope() {
    const int B = 8, HD = 128, NH = 32, NK = 8, R = 2064;
    std::vector<int32_t> pos{0, 1, 2047, 2048, 2300, 65535, 131071, 262143};
    Dev<int32_t> dp(pos);
    Dev<float> tab(B * HD);
    df2_rope_table(dp.p, B, 10000, tab.p, nullptr);
    std::vector<double> tr(B * HD);
    for (int r = 0; r < B; ++r)
        for (int i = 0; i < 64; ++i) {
            double a = pos[r] * std::pow(10000., -double(2 * i) / HD);
            tr[r * HD + i] = std::cos(a);
            tr[r * HD + 64 + i] = std::sin(a);
        }
    compare("rope_table", tab.get(), tr, 3e-7, 3e-6);
    auto q = randoms(B * NH * HD), k = randoms(B * NK * HD), v = randoms(B * NK * HD), gamma = randoms(HD);
    Dev<float> dq(q), dk(k), dv(v), dg(gamma), qo(q.size()), ko(k.size()), kr(R * NK * HD), vr(R * NK * HD);
    auto nr = [&](const std::vector<float> &x, int heads) {
        std::vector<double> out(x.size());
        for (int r = 0; r < B; ++r)
            for (int h = 0; h < heads; ++h) {
                int o = (r * heads + h) * HD;
                double ss = 0;
                for (int j = 0; j < HD; ++j)
                    ss += double(x[o + j]) * x[o + j];
                double sc = 1 / std::sqrt(ss / HD + 1e-6);
                for (int j = 0; j < HD; ++j) {
                    int i = j % 64;
                    int other = (j + 64) % 128;
                    out[o + j] = x[o + j] * sc * gamma[j] * tr[r * HD + i] +
                                 (j < 64 ? -1 : 1) * x[o + other] * sc * gamma[other] * tr[r * HD + 64 + i];
                }
            }
        return out;
    };
    df2_norm_rope_qk(dq.p, dk.p, dg.p, dg.p, tab.p, qo.p, ko.p, B, NH, NK, 1e-6f, nullptr);
    compare("norm_rope_q", qo.get(), nr(q, NH));
    compare("norm_rope_k", ko.get(), nr(k, NK));
    df2_kv_ring(dk.p, dv.p, dg.p, dp.p, tab.p, kr.p, vr.p, R, B, NK, 1e-6f, nullptr);
    auto ks = kr.get(), vs = vr.get();
    auto kn = nr(k, NK);
    std::vector<float> seen;
    std::vector<double> refs;
    for (int r = 0; r < B; ++r)
        for (int h = 0; h < NK; ++h)
            for (int j = 0; j < HD; ++j) {
                size_t at = ((size_t)h * R + pos[r] % R) * HD + j;
                seen.push_back(ks[at]);
                refs.push_back(kn[(r * NK + h) * HD + j]);
                seen.push_back(vs[at]);
                refs.push_back(v[(r * NK + h) * HD + j]);
            }
    compare("kv_ring", seen, refs);
    bench("rope_table", B * HD * 4, [&] { df2_rope_table(dp.p, B, 10000, tab.p, nullptr); });
    bench("norm_rope_qk", (q.size() + k.size()) * 8,
          [&] { df2_norm_rope_qk(dq.p, dk.p, dg.p, dg.p, tab.p, qo.p, ko.p, B, NH, NK, 1e-6f, nullptr); });
    bench("kv_ring", k.size() * 16,
          [&] { df2_kv_ring(dk.p, dv.p, dg.p, dp.p, tab.p, kr.p, vr.p, R, B, NK, 1e-6f, nullptr); });
}
void gemm() {
    for (auto shape : std::vector<std::pair<int, int>>{
             {2560, 640}, {2560, 1024}, {2560, 4096}, {4096, 2560}, {7680, 2560}, {12800, 2560}}) {
        int K = shape.first, M = shape.second;
        // Exact half scales make the weight decode independent of host half
        // support.
        std::uniform_int_distribution<int> dist(-127, 127);
        std::vector<uint8_t> w((size_t)M * K / 32 * 34);
        for (size_t b = 0; b < w.size() / 34; ++b) {
            w[b * 34] = 0;
            w[b * 34 + 1] = 0x24;
            for (int j = 0; j < 32; ++j)
                w[b * 34 + 2 + j] = (uint8_t)(int8_t)dist(rng);
        } // scale 1/64
        auto x = randoms(8 * K);
        Dev<uint8_t> dw(w), tile(w.size());
        Dev<float> dx(x), sd(8 * K / 32), out(8 * M);
        Dev<int8_t> q(8 * K);
        df2_q8_tile(dw.p, tile.p, tile.p + (size_t)M * K, K, (int64_t)M * K / 32, nullptr);
        df2_quant_rows(dx.p, K, 8, q.p, sd.p, nullptr);
        auto aq = q.get();
        auto ad = sd.get();
        std::vector<double> ref(8 * M);
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < 8; ++c) {
                double s = 0;
                for (int b = 0; b < K / 32; ++b) {
                    int dot = 0;
                    for (int j = 0; j < 32; ++j)
                        dot += (int8_t)w[((size_t)r * K / 32 + b) * 34 + 2 + j] * aq[c * K + b * 32 + j];
                    s += double(dot) / 64 * ad[c * K / 32 + b];
                }
                ref[c * M + r] = s;
            }
        for (int B : {1, 3, 8}) {
            Df2Mat mat{tile.p, out.p, M};
            df2_gemm(&mat, 1, q.p, sd.p, K, B, nullptr);
            auto got = out.get();
            compare("gemm vs CPU double", std::vector<float>(got.begin(), got.begin() + B * M),
                    std::vector<double>(ref.begin(), ref.begin() + B * M), 2e-4, 3e-5);
        }
        Df2Mat mat{tile.p, out.p, M};
        bench("gemm", w.size() + 8 * K * 4, [&] { df2_gemm(&mat, 1, q.p, sd.p, K, 8, nullptr); });
    }
}
void selector() {
    const int B = 7, V = 1024, K = 16, rank = 256;
    auto logits = randoms(B * V);
    for (int j = 0; j < 32; ++j)
        logits[j] = 10;
    Dev<float> dl(logits), tv(B * K);
    Dev<int32_t> ti(B * K);
    Dev<uint32_t> scratch(B * kDf2TopBlocks * K * 2);
    df2_topk(dl.p, V, nullptr, B, scratch.p, tv.p, ti.p, nullptr);
    auto ids = ti.get();
    auto vals = tv.get();
    for (int b = 0; b < B; ++b) {
        std::vector<int> ord(V);
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int c) {
            return logits[b * V + a] > logits[b * V + c] || (logits[b * V + a] == logits[b * V + c] && a < c);
        });
        for (int j = 0; j < K; ++j)
            if (ids[b * K + j] != ord[j] || vals[b * K + j] != logits[b * V + ord[j]])
                throw std::runtime_error("topk ties");
    }
    auto hp = randoms(B * rank, .1f);
    std::vector<uint16_t> cb(V * rank);
    std::vector<double> cbd(cb.size());
    auto f = randoms(cb.size(), .1f);
    for (size_t i = 0; i < cb.size(); ++i) {
        uint32_t u;
        std::memcpy(&u, &f[i], 4);
        cb[i] = u >> 16;
        u &= 0xffff0000;
        float v;
        std::memcpy(&v, &u, 4);
        cbd[i] = v;
    }
    Dev<uint16_t> dc(cb);
    Dev<float> dh(hp), op(B);
    Dev<int32_t> anchor(std::vector<int32_t>{17}), ot(B);
    df2_select(dh.p, tv.p, ti.p, dc.p, dc.p, rank, B, anchor.p, ot.p, op.p, nullptr);
    auto tok = ot.get();
    std::vector<double> probs(B);
    int prev = 17;
    for (int b = 0; b < B; ++b) {
        double sc[K];
        for (int j = 0; j < K; ++j) {
            sc[j] = vals[b * K + j];
            for (int d = 0; d < rank; ++d)
                sc[j] += cbd[prev * rank + d] * hp[b * rank + d] * cbd[ids[b * K + j] * rank + d];
        }
        int best = std::max_element(sc, sc + K) - sc;
        if (tok[b] != ids[b * K + best])
            throw std::runtime_error("selector path");
        double z = 0;
        for (double s : sc)
            z += std::exp(s - sc[best]);
        probs[b] = 1 / z;
        prev = tok[b];
    }
    compare("selector", op.get(), probs);
    bench("topk", B * V * 4, [&] { df2_topk(dl.p, V, nullptr, B, scratch.p, tv.p, ti.p, nullptr); });
    bench("selector", B * K * rank * 2,
          [&] { df2_select(dh.p, tv.p, ti.p, dc.p, dc.p, rank, B, anchor.p, ot.p, op.p, nullptr); });
}
void attention() {
    const int B = 8, NH = 32, NK = 8, D = 128, R = 2064, W = 2048;
    auto q = randoms(B * NH * D, .2f), kr = randoms(R * NK * D, .2f), vr = randoms(R * NK * D, .2f),
         kb = randoms(B * NK * D, .2f), vb = randoms(B * NK * D, .2f);
    Dev<float> dq(q), dk(kr), dv(vr), dbk(kb), dbv(vb), part((size_t)kDf2MaxChunks * 8 * 32 * 130), sd(B * NH * D / 32);
    Dev<int8_t> out(B * NH * D);
    Dev<int32_t> pos(1);
    for (int s : {0, 1, 2047, 2300}) {
        pos.put({s});
        df2_attention(dq.p, dk.p, dv.p, dbk.p, dbv.p, pos.p, R, W, part.p, out.p, sd.p, nullptr);
        std::vector<double> ref(B * NH * D);
        int lo = std::max(0, s - W + 1);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < NH; ++h) {
                int kh = h / 4;
                std::vector<double> scores;
                std::vector<const float *> values;
                for (int t = lo; t < s + B; ++t) {
                    if (t < s && s + b - t >= W)
                        continue;
                    const float *k = t < s ? &kr[((size_t)kh * R + t % R) * D] : &kb[((t - s) * NK + kh) * D];
                    const float *v = t < s ? &vr[((size_t)kh * R + t % R) * D] : &vb[((t - s) * NK + kh) * D];
                    double score = 0;
                    for (int d = 0; d < D; ++d)
                        score += q[(b * NH + h) * D + d] * double(k[d]);
                    scores.push_back(score / std::sqrt(double(D)));
                    values.push_back(v);
                }
                double mx = *std::max_element(scores.begin(), scores.end()), z = 0;
                for (auto &sc : scores) {
                    sc = std::exp(sc - mx);
                    z += sc;
                }
                for (size_t t = 0; t < scores.size(); ++t)
                    for (int d = 0; d < D; ++d)
                        ref[(b * NH + h) * D + d] += scores[t] / z * values[t][d];
            }
        quant_check("attention CPU double", out, sd, ref, NH * D, B);
    }
    bench("attention", (W + B) * NK * D * 8,
          [&] { df2_attention(dq.p, dk.p, dv.p, dbk.p, dbv.p, pos.p, R, W, part.p, out.p, sd.p, nullptr); });
}
} // namespace
int main(int argc, char **argv) {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            return 77;
        cu(cudaSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
        elementwise();
        taps();
        rope();
        gemm();
        selector();
        attention();
        cu(cudaDeviceSynchronize());
        std::puts("dflash2 kernels: PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
