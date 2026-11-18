# Paired-card long prefill

[简体中文](../zh/precision/paired-prefill.md)

With `--prefill-native-next`, a matching primary/first-peer pair can pipeline a
long prompt while an unlike third card remains an expert tier. This is a
whole-layer pipeline: it transfers residual rows between groups of layers,
rather than returning expert rows after each layer. The additional card retains
its existing decode and short-prompt roles.

## Selection and memory

The new selection applies to exactly two expert peers, a matching first pair,
and the 48-layer, 2,560-wide model. Existing two-card and all-alike-card selection
is retained. The measured paired layout keeps layers 0–25 on the primary and
26–47 on the first peer; the cost model prices this constraint before deciding
whether to split. Its stage image covers the wider range selected at startup.

With a 32,768-token cap, parts that fit within one configured prompt chunk stay
on the primary, and the paired path is eligible above 32,768 fresh tokens per
prompt part. Other chunk caps retain the two-chunk activation floor. This
counts newly read tokens, not the total cached context.
Images, active decoding slots, unsuitable links and insufficient memory retain
the existing decline/fallback rules. The prompt retains the primary-only
chunk boundaries. The final chunk finishes on the primary after the peer's
positional cells and running state return, and lent expert slots are restored
before decoding.

The temporary stage's hyper-connection matrices contain exactly the primary's
dequantized HC8 values. They are produced by the existing dequantizer at startup
and stored in the stage's existing BF16 image. Loading the original BF16 matrices
there would change the computation relative to the HC8-only primary.

## Checkpoints and interrupted reads

Periodic checkpoints merge the stages' running state. With DFlash2, they also
merge the final feature rows from all five tapped layers before saving the
drafter's state at that checkpoint position. The primary's feature relay uses
mapped pinned host memory; each peer has a temporary device buffer. At the
2,048-row window and this model's width, each feature buffer is 100 MiB. They
are released after the read. Admission accounts for the in-flight host feature
copies and retains the configured free-RAM floor.

A stopped read drains its workers and copies completed peer cells back before
releasing their temporary storage. Its checkpoints can then restore both the
running state and the corresponding positional cells. If this copy fails, new
checkpoints whose cells were not secured are discarded. Conversation file
formats, model identity and disk-cache policy are unchanged.

## Correctness checks

The machinery checks in this section use a two-chunk activation floor; the
one-chunk boundary has the separate coverage below.

Development checks use RTX 3090 Ti / RTX 3090 / RTX 3080 at enforced
400 / 320 / 250 W limits. For arithmetic comparisons, the tiers hold exactly
5,255 / 9,726 / 8,187 experts, adaptation is disabled, KV is int8 with its full
capacity resident, and the RAM expert copy is 20 GiB. This prevents elastic
cache resizing from moving experts between differently rounded execution paths.

Against the previous binary, six requests produced identical token IDs and
all 1,308 logged accepted-row logit hashes: a short prompt before/after the long
read, a 131,072-token prompt, cached/fresh 65,537-token branches, and a resumed
long prompt. Their continuations total 1,280 generated tokens. These are
bitwise checks of unchanged arithmetic; they are not an absolute BF16 quality
measurement at long context.

Focused tests cover feature-column assembly at 1, 33 and 2,048 rows, actual GPU
reads from the mapped buffer, both 320 × 10,240 and 10,240 × 320 HC8 matrix
shapes against a CPU reference, layer-range state transfers and selection
guards. Compute Sanitizer reports zero memory errors for the feature relay and
stage-weight tests. No production arithmetic kernel is added or changed.

Cancellation after 32,768 completed tokens, followed by a retry of the full
131,072-token prompt, also reproduces the baseline's tokens and all 262 logged
logit hashes. A disk-cache process restart reuses 131,065 tokens and reads seven;
its continuation and logit hashes match too. These integration checks use a
separate cache directory.

## One-chunk activation boundary

A warmed, within-process A/B/B/A diagnostic keeps the 32,768-token cap, layer
floor 26, fixed 30 GiB RAM expert copies, four live checkpoints, disabled
adaptation and zero prompt-prefix reuse. Disk caching remains enabled in an
isolated directory. Power limits are 400 / 320 / 250 W on the cards above.
These are agent-run development measurements.

| Fresh tokens in a prompt part | PP throughput change versus primary only |
|---|---:|
| 33,792 | +19.8% |
| 40,960 | +17.9% |
| 49,152 | +16.4% |
| 65,536 | +21.6% |

Below the configured cap, subdividing requests into additional pipeline chunks
does not establish a useful general improvement. The retained rule preserves
the primary-only chunk boundaries and does not introduce that subdivision.

A separate fixed-tier A/B/B/A check holds 6,077 / 9,717 / 8,192 experts, fully
resident int8 KV and 30 GiB RAM copies at the same power limits. At 32,769 fresh
tokens the greedy continuation and logit hashes differ; at 49,152 the text
matches but logit hashes differ. At 33,792 and 65,536 both match. The boundary
change is therefore not bitwise-neutral at every length, even though weight
formats and arithmetic kernels are unchanged. Greedy output can change.

The ordinary BF16 gate uses 100 contexts of 512 tokens, 25,500 scored rows,
int8 KV, full Q8_0 PLE, fixed 20 GiB RAM copies and disabled adaptation at the
same power limits. Initial expert placement is 7,669 / 9,331 / 7,576 in both
runs. Every scored row matches the previous measurement exactly.

| BF16 gate metric | Previous | One-chunk boundary |
|---|---:|---:|
| PPL | 3.30383093 | 3.30383093 |
| Mean KLD | 0.09661079 | 0.09661079 |
| Top-1 agreement | 89.69020% | 89.69020% |

This short-context gate does not exercise the cross-card path or establish
absolute BF16 quality at 32K context.

## Development performance

The standard `basalt-bench` comparison runs A/B/B/A, two repetitions per process,
with 30 GiB RAM expert copies, four live prompt checkpoints, disabled adaptation
and no prompt-prefix reuse. Disk conversation parking is disabled in these
benchmarks to isolate scheduling; production retains its disk-cache settings
and is checked separately. Initial expert placement is 7,669 / 9,331 / 7,576.
Power limits remain 400 / 320 / 250 W on the cards above. These are agent-run
development measurements, not maintainer headline results.

The nominal 128K and 240K fixtures render to 133,205 and 249,734 input tokens.
Rates below aggregate tokens over engine prompt time. Cold means the first
repetition in each fresh engine; warm means its second repetition.

| Input tokens | Run | Previous | Paired pipeline |
|---|---|---:|---:|
| 133,205 | Cold | 4,382 tok/s | 4,736 tok/s |
| 133,205 | Warm | 4,358 tok/s | 5,359 tok/s |
| 249,734 | Cold | 4,046 tok/s | 5,297 tok/s |
| 249,734 | Warm | 4,031 tok/s | 5,927 tok/s |

Cold runs have matching expert placement throughout KV growth: the primary
holds 6,962 experts at the intermediate depth and 6,326 at maximum KV backing.
On shrinking back for the warm repetition, the previous build regrows to
7,554 slots versus 7,549 with the pipeline's additional allocations. Warm
numbers therefore include that small placement difference; the fixed-tier
arithmetic checks above are separate.

The prose fixture generates 4,552 tokens in 2,048 verification windows
per build, including generation after the first long-context pass. Mean window
time is 21.899 ms before versus 21.663 ms with the pipeline, or 45.664 versus
46.162 windows/s. Generation rates are 101.50 versus 102.60 tok/s.

A separate short-only A/B/B/A check uses the same settings and initial expert
placement, four repetitions per process and no KV resizing. Its nominal
512 / 4K / 8K fixtures contain 549 / 4,191 / 8,353 input tokens. Warm rates
(excluding each process's first repetition) are 1,185.49 / 1,692.76 / 2,142.57
tok/s before versus 1,179.84 / 1,698.07 / 2,142.81 with the pipeline: changes
of −0.48% / +0.31% / +0.01%. Cold rates are 739.19 / 1,575.09 / 2,114.07
versus 728.79 / 1,568.63 / 2,114.07 tok/s. These small differences do not
establish a short-path speed improvement.

The best long run in this standard-fixture comparison is 5,973 tok/s. A separate
source-code-fixture screen, with the same power limits, 30 GiB RAM expert copies
and four checkpoints, reads an exact 32,768-token warm-up followed by fresh
131,072- and 245,760-token prompts. It measures 5,964 and 6,289 tok/s respectively,
with zero prompt-token reuse. This demonstrates over 6,000 tok/s after warm-up;
it does not establish that rate for every fixture or cold request.

A diagnostic Nsight run at 131,072 tokens uses 29 GiB RAM copies to leave room
for the profiler, with the same power limits. Primary and peer kernel-busy times
are 18.60 and 10.57 seconds. A 4.73-second gap before final verification contains
8.30 GB of primary host-to-device copies, identifying expert-cache restoration
as further work. A trial pinned upload ring does not improve the retained path
and is excluded.

A live-server check retains automatic expert/RAM sizing, adaptation and disk
parking. Fresh 131,072- and 245,760-token requests measure 4,762 and 5,585 tok/s;
another fresh 245,760-token request after warm-up measures 5,839 tok/s. Repeating
the compatible prompt reuses 245,753 tokens and reads seven in 166 ms, with the
same eight-token continuation. Production throughput therefore remains below
the isolated benchmark's peak with all of these settings active.
