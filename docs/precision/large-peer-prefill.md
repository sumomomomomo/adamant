# Large-chunk peer prefill prototype

[简体中文](../zh/precision/large-peer-prefill.md)

The diagnostic prototype distributes Flash-Next expert prefill between the
primary and first peer GPU for chunks of at most 4,096 uncached tokens. Larger
chunks retain ordinary primary-GPU expert computation. It is off by default.
Attention, routing, the shared expert, HC writes, vision and DFlash2 remain on the primary. Further expert tiers
retain their existing roles.

## Execution

The native fused path sends already quantized activations and the exact primary
row/group maps through portable pinned memory. The peer executes disjoint runs
of its resident experts with the existing kernels and returns unchanged F32 rows.
The primary omits those experts from its weight stream and fused launches, waits
for their rows, and runs the ordinary combine in the original routing order.
Layers outside fused format coverage retain their existing placement.
Quantization, arithmetic and reduction order are preserved.

The cutoff applies to each actual prefill chunk, independently of total conversation
length. Existing routed-only peer computation handles smaller chunks; the fused
extension applies only to its supported layers in streamed chunks up to the cutoff.
Long requests can therefore use the helper for a final chunk of at most 4,096 tokens.

Transfer buffers cover at most 4,096 tokens, and helper memory is reserved
before expert placement. Fused host buffers are planned before the RAM expert
copy. The prototype retains individual expert outputs; it does not combine peer
contributions locally or compress returned floats. Its transfer volume can
therefore outweigh the saved computation.

## Diagnostic comparison

`--prefill-peer-large 0` is the normal path. Mode `1` enables the prototype;
mode `2` allocates the same buffers with ordinary compute placement, for A/B
comparisons. Both diagnostic modes round peer room down to 256 MiB boundaries
before tier placement. Adaptation must be disabled and actual tier placement
checked when comparing greedy output or performance. These modes are development
diagnostics, not recommended serving settings.

## Validation

`prefill_fused_peer_test` compares returned expert rows byte for byte with the
existing primary-only fused kernels. It covers IQ4_XS, IQ3_XXS and IQ2_S gate/up with
IQ4_NL down, zero inputs, unused experts, fragmented expert ownership, no peer
ownership and full peer ownership, at 33, 257, 4,096 and 16,384 tokens. The
arithmetic kernels are unchanged; their CPU-reference tests remain applicable.


## Full-model development checks

On the mixed UD-IQ4_XS pack, 13 of 48 expert layers have fused format coverage.
The other 35 layers retain the ordinary large-chunk MMQ path on the primary.
Extending the old MMQ peer grouping did not preserve greedy identity in an
exploratory check, so that change is excluded from this prototype.

An A/B/B/A comparison on RTX 3090 Ti / RTX 3090 / RTX 3080 used power caps of
375 / 320 / 250 W, equal reserved buffers, fixed primary capacity, identical
peer placement and no adaptation or prompt-prefix reuse. All 24,576 decode
experts fit in VRAM. Prompt staging still includes RAM/file access and cold/warm
file-cache effects; these are development results, not headline benchmarks.
Rates below divide total tokens by total elapsed engine time across each pair.

The following measurements used the original unbounded prototype, before the
4,096-token cutoff was added.

| Prompt tokens | Ordinary compute, equal buffers | Unbounded prototype | Change |
|---|---:|---:|---:|
| 4,068 | 774.5 tok/s | 914.0 tok/s | +18.0% |
| 8,164 | 1,592.6 tok/s | 1,555.6 tok/s | -2.3% |
| 16,356 | 2,307.9 tok/s | 1,869.8 tok/s | -19.0% |

The 2,048-token generation check measured 162.3 versus 159.5 tok/s at the same
power caps. All five responses matched across all four runs: three text prompts,
a 2,048-token generation and an image prompt. Large-chunk logs confirm that the
peer returned computed expert rows. The prototype stays off by default.

With the prototype enabled, image follow-ups, RAM eviction, disk restores across
a process restart and image grammar-constrained decoding also passed, with
DFlash2 producing accepted proposals.

## Bounded-policy development checks

A separate A/B/B/A check of the bounded policy used the same power caps
(375 / 320 / 250 W), equal 4,096-token helper buffers, identical expert placement,
and no adaptation or prefix reuse. All seven fixtures matched across all four
runs: 512, 2,048, 4,096, 4,097, 4,098 and 16,384 prompt tokens, plus a
2,048-token generation fixture. No arithmetic kernels changed.

With prompt checkpoints disabled, the server handles the final prompt token
separately. A 4,097-token cold prompt therefore exercises a 4,096-row distributed chunk; 4,098 prompt tokens exercise
a 4,097-row primary-only chunk. Logs confirmed both sides of this boundary. Enabled prompt checkpoints can split
requests differently; the cutoff still applies to each actual chunk.

| Prompt tokens | Ordinary compute, equal buffers | Bounded prototype | Change |
|---|---:|---:|---:|
| 4,096 | 962.6 tok/s | 1,083.4 tok/s | +12.6% |
| 4,097 | 1,109.8 tok/s | 1,155.5 tok/s | +4.1% |
| 4,098 | 1,098.9 tok/s | 1,107.0 tok/s | +0.7% |
| 16,384 | 1,843.4 tok/s | 1,835.0 tok/s | -0.5% |

The 2,048-token generation fixture measured 200.1 versus 197.3 tok/s (-1.4%),
with identical 340 verification windows and 6.02 output tokens per window.
Average window time was 30.10 versus 30.53 ms; one prototype run was slower
(30.85 ms versus 30.21 ms). These development checks do not establish a
universal prefill gain or zero decode cost. Global defaults remain unchanged.
The bounded buffers reserve 443 MiB on the helper and 412 MiB of pinned host
memory, rounded down to whole MiB in the allocation log.

## Forced fused execution below the streamed threshold

A development A/B/B/A check compared ordinary routed-only MMQ against forcing
the streamed fused-peer path at 256, 512, 1,024 and 2,048 prompt tokens. Power
caps were 375 / 320 / 250 W. Each size used two prompts per process, with no
prefix reuse, prompt checkpoints or adaptation. All four processes held the
same 7,669 / 9,331 / 7,576 experts on their GPUs. Helper buffers and their
reservations were identical. All 24,576 decode experts fit in VRAM.

The experiment kept the ordinary MMQ helper available for the 35 uncovered
layers. Forcing streaming also changes staging, grouping, CPU fallback and
routing lookahead; this measures the complete forced path, not isolated fused
kernel throughput. RAM/file staging and its cache effects remain included.

| Prompt tokens | Routed-only MMQ | Forced fused peer | Change |
|---|---:|---:|---:|
| 256 | 471.6 tok/s | 153.9 tok/s | -67.4% |
| 512 | 894.8 tok/s | 307.1 tok/s | -65.7% |
| 1,024 | 1,133.9 tok/s | 576.5 tok/s | -49.2% |
| 2,048 | 1,419.5 tok/s | 983.8 tok/s | -30.7% |

Logs confirmed returned fused-helper rows for chunks of 255, 511, 1,023 and
2,047 tokens. One 512-token fixture and one 2,048-token fixture had stable
greedy-output differences: controls matched each other, and forced runs matched
each other. No BF16 PPL/KLD gate was run, so this experiment establishes no
quality equivalence for switching the small-chunk path.

A common 4,097-token prompt exercised the same 4,096-row fused prefill path in
both configurations before generating 2,048 tokens. Generation outputs matched;
rates were 170.9 versus 172.6 tok/s (+1.0%). The forced small-chunk path is not
retained. Routed-only sharing remains below the 2,560-token streamed threshold;
the fused extension remains limited to supported layers through 4,096 tokens.
