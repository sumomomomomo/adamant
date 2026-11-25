# RAM-backed prefill at long contexts

[简体中文](../zh/precision/long-context-prefill.md)

## Capacity and scheduling

The primary expert cache, elastic KV and prefill workspace share VRAM. A backed
cache tail sized for one configured prefill chunk per slot can run out as a
conversation grows beyond that chunk length. The scheduler then falls back to
256-token prefill chunks even when `--read-live-chunk` requests more.

For budgeted batch serving without a layer split, RAM backing now includes the
configured live chunk's workspace plus every primary slot's maximum elastic KV
growth (`max_cells`), including attached KV arenas. The backing region can extend
past the original prompt-only lending region. It is bounded by the GPU cache and
the resident RAM budget; it does not allocate maximum KV up front or increase the
budget. Backing and adaptive spare places are funded before ranked expert copies.
Insufficient RAM can still leave a shorter backed tail and reduce chunk sizes.
The current-ownership checks and adaptive backing protections remain in force.

`--read-beside` uses the existing background read path for shorter prompts too.
Without it, reads up to the 16,384-token threshold can hold decoding back. Sharing
GPU resources may lengthen the prefill itself while keeping output responsive.
This flag remains opt-in; the backing fix does not change the scheduling default.

## Development A/B

These are normal-allocation agent development checks, not headline benchmarks.
RTX 3090 Ti / RTX 3090 / RTX 3080 power limits were enforced at 400 / 320 / 250 W.
Both builds used two 262,144-token slots, int8 KV, MTP window 8, Q8_0 PLE,
32,768-token prefill capacity, the additional prefill peer, adaptation and a
21.5 GiB RAM budget. Prompt caching was enabled, disk conversation caching was
disabled, and each newly admitted prompt was uncached. Distinct synthetic prompt
prefixes prevented reuse across the measured requests. The candidate additionally
used `--read-beside`. These runs were before then after with automatic placement;
they do not isolate the effects of backing versus scheduling.

An approximately 80K-token prompt requested a long prose report. Once it began
decoding, a second slot received a new 9,622- or 19,279-token prompt. Decode rates
use actual per-slot generated counters sampled every 20 ms, from submission of
the second request until its first output. Streaming gaps use client content
events, which need not correspond one-to-one with tokens. Each case has one run.

| Metric | Before | After |
|---|---:|---:|
| Backed primary cache slots | 655 | 3,336 |
| Short prompt prefill tok/s | 1,001.2 | 511.1 |
| Short prompt prefill time | 9.61 s | 18.83 s |
| Decode during short prefill | 0.72 tok/s | 89.66 tok/s |
| Largest short-prefill output gap | 9.645 s | 0.072 s |
| Long prompt prefill tok/s | 294.1 | 598.5 |
| Long prompt prefill time | 65.54 s | 32.21 s |
| Decode during long prefill | 91.95 tok/s | 89.63 tok/s |
| Largest long-prefill output gap | 0.118 s | 0.155 s |

The candidate retained 1,024-token chunks during overlap. A logged growth step
displaced 407 experts, all with RAM copies; the equivalent old step had copies
for 118 and 287 routed experts among the remainder. Startup still placed all
experts on GPUs. This does not assert that all prefill file reads disappear.

Cancellation released the long decoder's slot between cases. The existing expert
source, budgeted loan restoration, RAM-home and repinning tests passed (4/4).
No arithmetic kernels changed. Different prefill chunking and RAM placement can
still change numerical results; the complete candidate quality run was skipped at the user's request. The old-build
100-chunk reference check completed (PPL 3.30281, mean KLD 0.0962279, top-1
89.7686%). No complete candidate quality result is claimed.
