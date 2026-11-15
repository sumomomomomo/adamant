# Small-prompt prefill investigation

[简体中文](../zh/precision/small-prefill.md)

The development target was 1,000 prompt tokens/s at 256 and 512 uncached tokens.
The tested system used RTX 3090 Ti / RTX 3090 / RTX 3080 at power caps of
375 / 320 / 250 W, a mixed UD-IQ4_XS pack, the full Q8_0 PLE table, int8 KV and
DFlash2. Short chunks retain routed-only MMQ and the first peer's expert share.

## Profile

Nsight Systems captured CUDA and OS runtime activity after warm-up for two
256-token and two 512-token prompts, each generating one token. Sampling and
CPU context-switch tracing were disabled. Profiling adds overhead; trace timings
are diagnostic observations, not headline throughput measurements.

The capture contained 76,119 CUDA kernel-launch calls, including 48,726
individual native expert-weight gathers. Kernel-launch API durations summed to
529 ms across threads. This is not a critical-path CPU overhead estimate.

| Prompt tokens | Host-to-primary traffic | GPU transfer activity duration |
|---|---:|---:|
| 256, first prompt | 5.122 GB | 434.6 ms |
| 256, second prompt | 5.146 GB | 433.5 ms |
| 512, first prompt | 7.711 GB | 681.1 ms |
| 512, second prompt | 7.723 GB | 674.1 ms |

Traffic was measured from each prompt's embedding broadcast to the next one's,
including the one-token decode tail and inter-request preparation. The primary
link was PCIe 4.0 x8. Effective transfer bandwidth was 11.3–11.9 GB/s. Transfers
can overlap kernels; their durations must not be added to kernel durations as
an end-to-end breakdown. At unchanged transfer volume, the 256 ms and 512 ms
targets require approximately 20.0 GB/s and 15.1 GB/s of host-to-primary traffic
respectively, before accounting for dependencies and other operations.

## Experiments

The comparisons fixed GPU expert placement at 7,669 / 9,331 / 7,576 entries and
RAM expert copies at 24 GiB. Prefix reuse, prompt checkpoints, parked caches and
adaptation were disabled. Arithmetic kernels and MMQ group membership were
unchanged. Existing K-quant MMQ tests passed against their CPU references;
grouped versus individual gathers were byte-identical.

The grouped-gather candidate batched both primary and first-peer weight gathers.
Its A/B/B/A comparison used two prompts per size per process. PP rates below use
total tokens divided by total engine prompt time across each configuration.

| Prompt tokens | Individual gathers | Grouped gathers | Change |
|---|---:|---:|---:|
| 256 | 510.1 tok/s | 516.5 tok/s | +1.3% |
| 512 | 726.2 tok/s | 724.6 tok/s | -0.2% |
| 1,024 | 920.5 tok/s | 918.7 tok/s | -0.2% |
| 2,048 | 1,225.7 tok/s | 1,228.0 tok/s | +0.2% |

A separate 2,048-token generation fixture measured 170.9 versus 173.9 tok/s.
Every generated response matched across the four runs.

Two subsequent A/B screens used CUDA batched transfer submissions, first for
resident pinned sources, then also for file-stager buffers. Neither justified
further performance claims: 256/512-token PP changes were -0.35%/+0.38% and
-0.23%/-0.28%, respectively. All generated outputs matched. The pinned-source
screen generated 2,048 tokens at 172.8 versus 172.6 tok/s; the all-host screen
generated 512 tokens at 139.3 versus 138.4 tok/s. These are separate workloads.

## Additional expert peer

A separate prototype let another resident expert tier compute routed-only
prefill rows, alongside the existing first peer. Each peer used the existing
MMQ transport and kernels, returned its own contiguous unweighted row block,
and joined before the original per-token combine. Streamed chunks and the
bounded fused-peer path were unchanged.

This development A/B/B/A used power caps of 400 / 320 / 250 W and 20 GiB of
RAM expert copies. Both configurations allocated the additional peer's buffers;
only the candidate dispatched work to it. GPU placement stayed at
7,669 / 9,331 / 7,576 experts, with all experts available on GPUs for decode.
The extra buffers fit without displacing experts. Reuse, parking and adaptation
were disabled. These rates are not directly comparable with the earlier
24 GiB RAM-copy investigation.

| Prompt tokens | First peer only | Both peers | Change |
|---|---:|---:|---:|
| 256 | 460.0 tok/s | 953.2 tok/s | +107.2% |
| 512 | 662.3 tok/s | 1,269.1 tok/s | +91.6% |
| 1,024 | 866.8 tok/s | 1,469.6 tok/s | +69.5% |
| 2,048 | 1,198.5 tok/s | 1,625.0 tok/s | +35.6% |

Each length had four measurements per configuration. First-layer dispatch logs
confirmed 775–787 rows on the additional peer for 256-token requests and
1,310–1,313 rows for 512-token requests. The final prompt token uses the decode
path, so the logged batched lengths were 255 and 511.

All 18 candidate responses matched the control's greedy text, including two
2,048-token generation runs. A separate original-binary run matched all eight
short-prompt controls at the same 20 GiB RAM budget.
Generation measured 170.2 versus 172.8 tok/s;
there was no observed decode penalty. The existing transport parity test on
the primary/additional-peer pair passed its main-GPU and CPU-reference
tolerances. No arithmetic CUDA kernels, weight formats or combine order changed.

## Result

The scheduling-only experiments did not reach either short-prompt target.
Computing another resident tier's experts exceeded the 512-token target and
approached the 256-token target in the separate controlled test. This is
development evidence, not a broad quality or workload validation. The prototype
engine changes were reverted after that test. The later
[daily-use hardening](extra-peer-prefill.md) records the bounded additional-tier
path, memory fallback and full BF16 quality comparison.
