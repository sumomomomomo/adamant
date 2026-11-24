# Experimental three-GPU prefill cutoffs

[简体中文](../zh/precision/peer-prefill-cutoffs.md)

Serving with `--prefill-native-next`, an additional computing expert
peer, two expert tiers, comparable primary/first-peer cards and non-exclusive
expert placement uses the measured cutoffs below. Other configurations retain
their existing policy. These features remain experimental and off by default.

The routed policy also applies with batch slots. The additional peer has its own
prefill transport buffers; its device allocation is reserved before filling its
expert tier. Before a prompt starts, outstanding adaptive workers finish. During
an overlapped read, expert adaptation stays paused while decoding and prefill
read the shared expert weights. Short reads retain the existing scheduling:
up to 16,384 tokens run alone unless `--read-beside` is enabled.

The fused first-peer helper (`--prefill-peer-large`) still requires single-slot
serving. It is not needed for the tuned routed policy.

| Actual chunk tokens | Previous policy | Tuned policy |
|---|---|---|
| Begin streamed fused prefill | 2,560 | 6,144 |
| Additional peer's largest routed chunk | 2,048 | 6,143 |
| Fused first-peer helper's largest chunk | 4,096 | 4,096 |

Below the streamed threshold, MMQ computes routed experts across the primary and
both peers. Above it, non-resident experts stream to the primary. The fused helper
has no eligible chunk in the tuned configuration because its upper bound is below
the streamed threshold; its unused device buffers are omitted. The original
helper policy remains available in configurations outside this experimental mode.
The limits apply to actual chunks, which can be shorter than the full prompt due
to checkpoints and the separate final token.

No arithmetic kernels, model weights, PLE precision, KV formats or decode policy
change. Mid-size prompts switch between existing MMQ and fused arithmetic, and
buffer reservations can change expert placement. Numerical differences and
changed greedy responses are possible.

## Batch-slot serving validation

A normal-allocation A/B/B/A development comparison used one slot in the previous
build and two full-context slots with the additional peer enabled. Both used MTP
window 8, int8 KV, 262,144 context per slot, Q8_0 PLE, 23 GiB resident RAM experts,
and RTX 3090 Ti / RTX 3090 / RTX 3080 at 400 / 320 / 250 W. Expert adaptation
was disabled. Each arm ran three requests per size; all requests had zero cached
tokens. These are configuration comparisons: the slots and buffer reservations
change expert placement, so they do not isolate kernel performance.

The unaccelerated two-slot column is a separate three-request median with the
same prompts, cache settings and power limits. It retains ordinary first-peer
prefill but has no additional peer or tuned routed cutoffs.

| Actual prompt tokens | One slot, accelerated | Two slots, unaccelerated | Two slots, accelerated |
|---|---:|---:|---:|
| 1,120 | 1,257 | 982 | 1,430 |
| 2,208 | 1,450 | 1,428 | 1,558 |
| 4,400 | 1,548 | 1,143 | 1,611 |
| 6,576 | 1,630 | 1,667 | 1,653 |
| 8,768 | 2,135 | 2,146 | 2,130 |

All rates are tok/s.

An overlapping 23,491-token read completed while another slot generated 2,048
tokens. That decode ran at 211.6 tok/s; its largest observed streaming gap was
0.294 s. The prompt's 256-token chunks exercised the additional peer while 258
decode windows ran. A disconnected stream released its slot and a subsequent
uncached request completed. These checks validate progress and cleanup, not
universal latency bounds. Peer transport parity, all four prefill layout tests,
the cutoff policy test and fused-peer parity passed.
The overlap and disconnect checks also passed with normal expert adaptation
enabled; two simultaneous requests were observed decoding in both slots.

A separate single-pass `basalt-bench` comparison used 20 GiB resident RAM experts
and normal allocation at the same power limits. Its 1,069 / 2,110 / 4,191-token
prefills measured 1,248 / 1,542 / 1,649 tok/s with one slot and
1,170 / 1,529 / 1,647 tok/s with two. The cold 6,272 / 8,353-token fixtures were
slower: 1,615 / 2,104 versus 1,400 / 1,976 tok/s. The repeated serving results
above do not establish a gain for every prompt or cold allocation.

| Decode fixture | One slot, ms/window | Two slots, ms/window | One / two slots, passes/s |
|---|---:|---:|---:|
| conv | 20.898 | 21.158 | 47.851 / 47.264 |
| struct | 34.094 | 34.464 | 29.331 / 29.016 |
| prose | 19.829 | 19.838 | 50.431 / 50.410 |

The full BF16 gate used 100 chunks, 25,500 scored positions, forced window T=3
and 20 GiB resident RAM experts at the same power limits. Arithmetic kernels are
unchanged, but slot storage and peer buffers change tier placement and can alter
greedy output.

| Metric | One slot | Two slots |
|---|---:|---:|
| PPL | 3.30806 | 3.30862 |
| Mean BF16 KL divergence | 0.0965205 | 0.0966209 |
| BF16 top-1 agreement | 89.7176% | 89.6510% |

## Development comparison

All results in this section use RTX 3090 Ti / RTX 3090 / RTX 3080 at enforced
400 / 320 / 250 W limits, int8 KV, full Q8_0 PLE and DFlash2. They are agent-run
development measurements, not headline benchmarks.

A screening compared streamed thresholds of 2,560, 4,096, 6,144 and 8,192, and
an extended 8,192-token fused helper. All arms used identical reserved peer
buffers, the same 7,669 / 9,331 / 7,576 initially placed experts, fixed 20 GiB
RAM copies, disabled adaptation and no prompt-prefix reuse. Two source-code
prompts per size followed three long-prompt warmups. Rates divide total tokens
by total engine time.

| Exact prompt tokens | Previous policy, tok/s | Tuned policy, tok/s |
|---|---:|---:|
| 2,048 | 1,619.8 | 1,602.8 |
| 3,072 | 1,100.2 | 1,647.8 |
| 4,096 | 1,314.7 | 1,670.9 |
| 5,120 | 1,347.1 | 1,673.2 |
| 6,144 | 1,598.5 | 1,682.5 |
| 8,192 | 2,087.8 | 2,050.8 |
| 16,384 | 3,714.4 | 3,662.5 |
| 32,768 | 4,483.2 | 4,416.0 |

The 8,192-token routed threshold did not improve 8K throughput. Extending the
fused helper to 8,192 reduced its 8K result to 1,767.7 tok/s. The 512-token decode
check used the same 157 verification windows in every arm: previous versus tuned
24.23 / 24.06 ms per window, or 134.6 / 135.5 tok/s. Small changes in long-prompt
and decode timings require an alternating comparison rather than a zero-cost claim.


An A/B/B/A check with the standard `basalt-bench` fixtures, fixed reservations
and the same expert placement measured 4,191-token prompts at 1,113.3 versus
1,598.7 tok/s (+43.6%). Its prose fixture generated the same 1,072 tokens through
472 windows in every process: 22.122 versus 22.409 ms/window (45.20 versus
44.63 passes/s), a 1.3% slower diagnostic decode result. This is not evidence of
zero decode cost; the normal serving allocation is checked separately.

## Reproduction and validation

`--diag-prefill-cutoffs STREAM,EXTRA,LARGE` is an A/B diagnostic, not a serving
recommendation. It accepts streamed/helper bounds through 8,192 and an additional
peer bound through 6,144, strictly below STREAM. Every diagnostic arm reserves
8,191 first-peer routed tokens, 6,144 additional-peer tokens and 8,192 fused-helper
tokens, so changing the selected cutoffs does not change those reservations.
Actual tier placement must still be checked. Normal serving sizes buffers to its
selected policy. The diagnostic is off by default.

`prefill_policy_test` checks configuration eligibility, cutoff validation and
reservation bounds. The four `prefill_layout_*` tests cover buffer layouts around
the thresholds. `prefill_fused_peer_test` checks byte-identical returned rows with
fragmented, empty and full peer ownership. No new arithmetic kernel is introduced.

The BF16 reference has 512-token contexts and scores the decode half after a
256-token prefix. Forced streamed arithmetic and ordinary routed arithmetic can
be compared there. It does not directly measure BF16 quality at a 4K or 6K context.

## BF16 quality

At the same 400 / 320 / 250 W limits, the streamed control reproduced all 5,100
rows of the first 20 chunks of the existing 100-chunk measurement byte for byte.
The routed candidate scored all 100 chunks through `basalt-bench --quality`.
These comparisons used the fixed-reservation diagnostic.

| Metric, 25,500 scored positions | Streamed control | Routed candidate |
|---|---:|---:|
| PPL | 3.30826788 | 3.30806771 |
| Mean BF16 KL divergence | 0.09611791 | 0.09615407 |
| BF16 top-1 agreement | 89.76471% | 89.71765% |

PPL changed -0.00605%, mean KLD +0.00003615 and top-1 agreement -0.04706
percentage points. The measured differences are small; they do not establish
bitwise equivalence or actual 4K-context BF16 parity.

## Normal serving validation

A separate A/B/B/A comparison used the original binary and the final automatic
policy with their normal buffer allocations. It kept the same initially placed
7,669 / 9,331 / 7,576 experts, fixed RAM copies and disabled adaptation. Each
process ran the standard prose fixture three times. Power limits remained
400 / 320 / 250 W.

Both modes generated the same per-run token and verification-window counts.
Across six requests per mode, the original versus tuned build measured
20.81377 / 20.57810 ms per verification window and 109.1329 / 110.3827 tok/s.
The normal-allocation comparison does not reproduce the diagnostic decode
slowdown above; it measured a 1.13% reduction in window time.

The final automatic policy also ran all 100 BF16-reference chunks with normal
allocations: PPL 3.30383093, mean KLD 0.09661079 and top-1 agreement 89.69020%.
All 25,500 rows matched the previously validated ordinary routed-prefill
reference exactly. The fixed-reservation diagnostic used a different peer
placement during its quality run, so its row values are not interchangeable
with this serving measurement.
