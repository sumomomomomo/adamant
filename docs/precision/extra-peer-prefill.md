# Additional expert-tier prefill

[简体中文](../zh/precision/extra-peer-prefill.md)

The additional expert tier can compute short-prompt rows alongside the primary
GPU and the first expert peer. Validation covers the mixed UD-IQ4_XS pack with
expert ablation enabled, the full Q8_0 PLE table, int8 KV and DFlash2 on Linux
with RTX 3090 Ti / RTX 3090 / RTX 3080 20 GB cards.

With native-next and two computing expert tiers, the [experimental three-GPU
cutoffs](peer-prefill-cutoffs.md) extend this routed range. The limits below
describe the ordinary additional-peer path.

## Dispatch and memory

The additional tier is enabled explicitly by the serving configuration; it is
off by default. Its routed-only chunk range is **128–2,048 rows**, inclusive.
Smaller cached tails and larger chunks retain the existing paths. The first
peer's routed-only path, bounded fused extension and long-chunk primary path
keep their previous limits. Limits apply to actual uncached chunks, not total
conversation length or the API's prompt count.

CPU expert selection is performed before assigning additional GPU work. Each
peer returns its own contiguous, unweighted row block. All peers join before
the original per-token combine. Arithmetic kernels and weight formats are
unchanged, but changing MMQ launch geometry and device work partitioning can
change floating-point rounding and greedy text. This is not universally
bitwise-neutral.

Additional buffers use spare memory after the decode expert tiers are filled.
Admission leaves at least 64 MiB on the primary, 128 MiB on the added peer and
64 MiB beyond the host RAM reserve. Allocation failure or insufficient memory
disables that optional peer with a warning; experts are not evicted to fit it.
An abandoned expert operation drains outstanding peer work before its buffers
can be reused. Successful requests use stream-event joins.

## Development measurements

The CPU-selection-preserving path was compared A/B/B/A at **400 / 320 / 250 W**,
with 20 GiB of RAM expert copies, fixed placement of 7,669 / 9,331 / 7,576
experts, and prefix reuse, parking and adaptation disabled. Controls allocated
the same additional buffers without dispatching work. Each prompt length had
four measurements per configuration.

| Prompt tokens | First peer only | Additional peer | Change |
|---|---:|---:|---:|
| 256 | 460.8 tok/s | 897.9 tok/s | +94.9% |
| 512 | 660.7 tok/s | 1,242.7 tok/s | +88.1% |

A 2,048-token generation fixture measured 173.76 versus 173.59 tok/s; its text
matched across configurations. Both configurations retained all experts on the
GPUs for decode. These are development measurements, not general hardware
claims. Normal serving also pays for checkpoints and may use different tiers.

Post-deployment smoke checks kept the normal four checkpoints, disk parking and
adaptation enabled. Three fresh source-code prompts per length, including the
first cold request, measured median PP of 610.4 tok/s at 256 and 907.0 tok/s at
512. These use different prompts and serving settings from the controlled
comparison and are not a paired speedup measurement.

## BF16 quality comparison

Expanded greedy checks found differences, so quality was measured against the
[published BF16 reference](https://huggingface.co/datasets/jesdga/basalt-kld-bases)
at revision `a65bae55f703e9c6dab6e9f335146c532fb7dd95`. The reference is
unsloth BF16 through llama.cpp `436f6f89e`. Its downloaded SHA256 matched the
published checksum.

Both the preceding build and the candidate ran the full 100 chunks, scoring
25,500 positions through three-token decode windows. Power limits, GPU expert
placement, 20 GiB RAM copies and the model were matched; adaptation and prefix
reuse were disabled. The candidate's additional peer processed the 256-row
prefills. Scores include the existing model quantization and ablation.

| Metric | Preceding build | Additional peer |
|---|---:|---:|
| PPL | 3.300916 | 3.303831 |
| Mean KLD | 0.096594 | 0.096611 |
| Top-1 agreement with BF16 | 89.6549% | 89.6902% |

PPL changed by +0.0883%, mean KLD by +0.0000167 and top-1 agreement by
+0.0353 percentage points. The approximate 95% interval for paired chunk-mean
PPL change was -0.181% to +0.359%; this comparison did not establish a clear
quality regression. Greedy wording can still differ.

## Integration coverage

The existing helper transport test passed its primary-GPU and double-precision
CPU tolerances on the primary/third-card pair. Draining a request without a
normal join and subsequently reusing its buffers produced byte-identical rows.
Three conversation-cache CPU tests and 103 selected server tests passed.

A real VRAM-pressure test held 5 GiB on the third card and constrained the peer
reserves: the optional helper was disabled and grammar-constrained inference
still succeeded. Normal serving checks covered image changes, image history
with DFlash2, old-build disk caches, tiny cached continuations, a 32K history
append, and cancellation followed by another request.

With adaptive placement and elastic KV enabled, restart checks assess state
integrity, reused tokens, remembered content and continued drafting. Exact
prose equality is not a valid byte-identity check for that changing placement.
Use fixed tiers and the quality gate for controlled numerical comparisons.

A five-minute mixed soak completed 83 cached requests plus 24 concurrent cold
prefills, with added-peer dispatch verified on every cold request. Model work
remained serialized by the server. Engine threads stayed at 304; RSS and GPU
allocations showed no accumulating growth after warm-up. New-build disk
checkpoints also restored successfully in the preceding build for rollback.
