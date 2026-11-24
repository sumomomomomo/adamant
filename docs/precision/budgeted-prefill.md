# Budgeted RAM backing for overlapping prefill

[简体中文](../zh/precision/budgeted-prefill.md)

With batch slots and an explicit resident RAM budget, prompt workspace borrows a
contiguous tail of the primary GPU expert cache. The tail's weights must have RAM
copies so they can be restored without file reads while another slot decodes.

## Selection and capacity

The model sizes backing for a 1,024-token prompt chunk plus the slots' estimated
KV growth through one configured prefill chunk. This region is bounded by the
normal prompt lending region. The budgeted loader funds adaptive spare places
and the backed tail first, then spends the remaining budget on ranked experts.
Copies and spare places together stay within the requested RAM budget.

Only complete experts in a contiguous suffix count as backed. An expert that
does not fit ends that suffix; variable expert sizes are accounted individually.
The existing non-budgeted whole-complement path is unchanged.

`--read-live-chunk` also caps the background overlap path (default 1,024 tokens).
KV growth reduces the remaining loan capacity and can select smaller chunks.
The scheduler checks current RAM ownership before lending. Adaptive promotions
cannot replace a protected tail weight with an expert that lacks a RAM copy.
Short reads retain the existing run-alone scheduling unless `--read-beside` is on.

This changes RAM selection, not weight formats or arithmetic kernels. Selection
and chunking can change which CPU/GPU arithmetic runs and can alter greedy output.
It does not remove long-context admission or workspace-setup stalls.

## Development validation

Normal-allocation serving runs used RTX 3090 Ti / RTX 3090 / RTX 3080 at enforced
400 / 320 / 250 W, int8 KV, MTP window 8, Q8_0 PLE, two 262,144-token slots,
22.9 GiB resident RAM budget, 32,768-token prefill capacity and the additional
prefill peer. The embedding service ran on CPU. These are agent development
measurements, not headline benchmarks or isolated kernel timings.

Two candidate requests were bracketed by four old-build requests. Each used the
same uncached 21,305-token prompt while another slot generated 4,096 tokens.

| Metric | Before | After |
|---|---:|---:|
| Ordinary overlap chunk | 256 tokens | 1,024 tokens |
| Prefill time range | 31.30–34.46 s | 22.72–26.25 s |
| Median prefill time | 33.34 s | 24.48 s |
| Median concurrent decode time | 18.22 s | 18.38 s |
| Largest streaming gap, per-run range | 0.256–0.691 s | 0.414–0.462 s |

Median prefill latency fell 26.6%. The decode difference is small relative to
these few runs' variation; this does not establish zero decode cost.

A separate paired stress request gave both slots roughly 21K-token prompts. The
candidate used 768-token chunks as KV storage consumed cache space. Its second
prefill took 25.41 s versus 33.29 s before. Both builds retained long-context
setup stalls: largest observed stream gaps were 5.61 s after and 5.14 s before.
Both requests completed. The candidate's logged KV evictions had RAM copies and
startup reported no file-only experts.

`file_expert_budget_lend_test` checks variable-size experts, partial budgets,
spare places, pinned and pageable copies, duplicate/invalid profile entries, and
unchanged whole-complement selection. It overwrites borrowed GPU slots and
verifies byte-identical restoration from RAM. The existing source, RAM-home and
repinning tests pass. Live checks cover cancellation, recovery and two concurrent
decoders with normal adaptation enabled.

The standard single-active-slot benchmark used the same normal-allocation setup
and power limits, with adaptation disabled. These sequential development runs
are supporting checks, not a pinned-placement performance claim.

| Decode fixture | Before ms/window (passes/s) | After ms/window (passes/s) |
|---|---:|---:|
| Conversation | 21.0796 (47.4393) | 21.1587 (47.2620) |
| Structured | 34.3183 (29.1390) | 34.3670 (29.0977) |
| Prose | 19.7031 (50.7535) | 19.7237 (50.7004) |

Uncached standalone prefill measured 1,565.32 → 1,666.00 tok/s at 4K and
2,134.20 → 2,240.68 tok/s at 16K under those power limits.

The full BF16-reference decode quality gate used 100 chunks / 25,500 rows,
MTP window 5, the same RAM budget and GPU power limits, and disabled adaptation.
Normal allocation was retained, so placement and arithmetic may differ.

| Quality metric | Before | After |
|---|---:|---:|
| PPL (BF16 reference 3.27327) | 3.30101 | 3.30548 |
| Mean KLD | 0.0969211 | 0.0964061 |
| Top-1 agreement | 89.7647% | 89.8471% |

The small PPL increase is within the reported roughly 0.043 sampling error;
KLD and top-1 show no corresponding degradation. Numerics are not bitwise equal.

Deployment smoke validation used a lower 22.0 GiB budget because the live server's
RAM guard allowed less headroom than the benchmark process. Both slots served
requests, overlap used 1,024-token chunks, and KV-displaced experts had RAM copies;
startup and the metrics endpoint reported zero SSD spill.
