# Long prefill with native expert weights

[简体中文](../zh/precision/long-prefill.md)

The off-by-default diagnostic `--prefill-native-next` enables experimental Q5_0
down projections and v2 IQ4_XS gate/up. It is not a recommended serving setting.
The normal path retains its existing kernels and thresholds. With the diagnostic
enabled, the native fused prompt path covers Q5_0 as well as Q2_0 and IQ4_NL. On the mixed UD-IQ4_XS pack, this extends fused coverage from 13 to all
48 expert layers. The larger fully fused chunks need less temporary memory than
the MMQ fallback, allowing a 32,768-token chunk on the measured machine.

## Execution and precision

Q5_0 blocks are decoded directly to their signed integer codes and original
scales. The v2 down kernel reads raw blocks into shared memory and decodes them
into MMA fragments. IQ4_XS gate/up uses the same v2 schedule: each raw unit holds
128 values' codes, with a separate copy of the super-block scale metadata.
IQ3_XXS gate/up and IQ4_NL down retain their existing v2 implementations.

The streamed ring holds 384 expert blobs for chunks of at least 16,384 tokens,
including fully fused chunks. Smaller streamed chunks retain 256 slots. This
keeps weight transfers working through the longer attention stages.

Weights and their quantization, the full Q8_0 PLE table, and the int8 KV format
are unchanged. Fused activation quantization and the v2 accumulation order can
differ from the old MMQ path. This is a numerical change; bitwise logits and
identical greedy responses are not promised.

The large expert helper still has a 4,096-row cutoff. Its experimental format coverage
includes Q5_0 when the diagnostic is enabled. Longer reads can use the
[paired-card pipeline](paired-prefill.md), which assigns complete layers to each card.
The additional helper uses the [experimental three-GPU cutoffs](peer-prefill-cutoffs.md)
when both expert tiers compute routed rows. Three-GPU decode, DFlash2, vision and
conversation-cache serialization retain their existing paths.

## Validation method

Kernel checks compare random native blocks, including negative Q5_0 scales,
against the previous MMQ path and a CPU reference with double accumulation.
They cover zero inputs, skewed routing, unused experts, unaligned blobs, and
different expert ranges. Timing includes the real 2,560/640 dimensions, 512
experts, top 10, and chunks through 32,768 tokens. Peer transport also checks the
new format pairs against primary-only expert output.

The BF16 quality reference has 100 contexts of 512 tokens and 25,500 scored
positions. The diagnostic `--diag-prefill-long` requires `--ppl-base` and forces
the streamed arithmetic onto its 256-token prefixes; normal serving thresholds
are unaffected. This exercises the changed arithmetic through the real decode
quality gate. It is not a 32K-context BF16 reference measurement.

Development serving comparisons use alternating old/new/new/old runs, identical
expert placement, fixed RAM copies, disabled adaptation and prompt reuse, and
400/320/250 W limits on RTX 3090 Ti / RTX 3090 / RTX 3080. Cold and warm requests
are recorded separately; reported steady rates exclude warm-up requests.

The benchmark and quality results below are development measurements on that
machine, not a general performance claim.

## Development results

The A/B/B/A serving comparison used 32,768 uncached input tokens, int8 KV, full
Q8_0 PLE, fixed 20 GiB RAM expert copies, and 7,669 / 9,331 / 7,576 resident
experts. Power limits were 400 / 320 / 250 W on the three cards above. The old
build used a 16,384-token chunk cap; the experimental build used 32,768. Three
warm-up requests preceded two measured long requests per process. This measures
warmed weight/file caches without token-prefix reuse, not cold-start throughput.

| Engine metric | Old | Experimental | Change |
|---|---:|---:|---:|
| 32,768-token prefill | 3,714.7 tok/s | 4,510.9 tok/s | +21.43% |
| 512-token prefill | 1,073.7 tok/s | 1,058.4 tok/s | -1.43% |
| 4,097-token prefill | 1,113.5 tok/s | 1,277.8 tok/s | +14.76% |
| 2,048-token generation after 512-token prompt | 120.83 tok/s | 119.60 tok/s | -1.02% |

All four generation responses matched, with 717 verification windows and
1,331 accepted proposals out of 1,865. Mean window times were 23.43 / 23.69 /
24.08 / 23.85 ms in A/B/B/A order (42.68 / 42.21 / 41.53 / 41.93 windows/s).
These runs do not establish zero decode cost or the 5,000 tok/s target. A separate
A/B/B/A decode check with the same 16,384-token cap on both builds measured
118.68 versus 117.63 tok/s (-0.89%), again with identical responses. All four
had 744 windows and 1,305 accepted proposals out of 1,873; their window times
were 23.02 / 23.14 / 23.66 / 23.37 ms. GPU power limits remained 400 / 320 / 250 W.
The first attempted repeat overlapped a CPU build and is excluded from this
comparison.

A separate settings-only check on the old binary with a 32,768-token cap
measured about 3,990 tok/s after warm-up. The guarded experimental rebuild
repeated about 4,514 tok/s. These single-process checks corroborate the result;
they are not the alternating comparison above. Global defaults remain unchanged.

At the same power limits, the IQ4_XS/Q5_0 expert microbenchmark at 32,768 tokens,
2,560/640 dimensions, 512 experts and top 10 measured 49.228 ms for MMQ versus
35.528 ms for the new v2 kernels. The logical gate/up/down work is 3.221 TFLOP
per layer: about 65.4 versus 90.7 effective TFLOP/s. This excludes attention,
dense work and weight transfers. The changed CUDA translation unit compiled in
6.43 seconds with `nvcc --time` on the development host.

All native CPU/MMQ reference tests, peer byte-identity tests and the batch-width /
token-position invariance checks passed. Compute Sanitizer reported zero memory
errors for the IQ4_XS test family. Unchanged ordinary short-path quality rows
matched the prior build exactly over 5,100 scored positions.

The new v2 reference pairs report the following maximum absolute / pointwise
relative errors on random inputs (256 tokens, 64 experts, relative denominator
floor `1e-6`; power limits 400 / 320 / 250 W):

| Gate/up + down | vs CPU max abs | vs CPU max rel | vs MMQ max abs | vs MMQ max rel |
|---|---:|---:|---:|---:|
| IQ3_XXS + Q5_0 | 53.3757 | 33,678.1 | 3.15881 | 26.7957 |
| IQ4_XS + Q5_0 | 14.8402 | 48,717.3 | 1.40591 | 23.5764 |

Near-zero reference values inflate pointwise relative errors. Relative RMS
versus the CPU is 0.01440 / 0.01455, comparable to MMQ; v2 versus the earlier
fused implementation is 0.00005942 / 0.00004451. Full-model quality is evaluated
separately below.

## Quality limits

Both columns below force streamed arithmetic onto the same BF16 reference's
256-token prefixes, with the same expert placement, int8 KV, full Q8_0 PLE,
20 GiB RAM copies and 400 / 320 / 250 W power limits. Each scores all 100 chunks
(25,500 positions). The old arithmetic column is a diagnostic control built
from the previous kernels; it avoids confounding CPU versus streamed prefill.

| Metric | Old streamed arithmetic | Experimental streamed arithmetic |
|---|---:|---:|
| PPL | 3.30193750 | 3.30826788 |
| Mean BF16 KL divergence | 0.09665480 | 0.09611791 |
| BF16 top-1 agreement | 89.70196% | 89.76471% |

PPL increased 0.1917%; KL divergence and top-1 agreement improved. An exploratory
0.1% absolute PPL drift tolerance was not met. This tolerance is a development
choice, not an established project threshold. A paired chunk estimate puts the
approximate 95% interval for PPL change at -0.070% to +0.454%; these measurements
do not establish a general quality loss or equivalence. They also do not measure
BF16 parity at actual 32K context. The kernels remain diagnostic rather than
replacing the normal path.

Long-chunk expert splitting, peer-local summation, sparse expert launches,
smaller pipelined helper tiles, a 512-slot ring and larger expert launch batches
did not improve the retained end-to-end result. Those prototypes are excluded.
Automatic RAM growth under reduced staging reservations caused memory pressure;
that planner change is also excluded. Larger prompt chunks can delay cancellation
because the existing cancellation check occurs at chunk boundaries.
