# Experimental three-GPU prefill cutoffs

[简体中文](../zh/precision/peer-prefill-cutoffs.md)

Single-slot serving with `--prefill-native-next`, an additional computing expert
peer, two expert tiers, comparable primary/first-peer cards and non-exclusive
expert placement uses the measured cutoffs below. Other configurations retain
their existing policy. These features remain experimental and off by default.

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
