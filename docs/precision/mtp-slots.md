# Two-slot MTP window sweep

[简体中文](../zh/precision/mtp-slots.md)

These are agent development measurements on RTX 3090 Ti / RTX 3090 / RTX 3080
at enforced 400 / 320 / 250 W, using int8 KV, Q8_0 PLE, two full-context slots,
a 22.0 GiB resident budget and additional-peer routed prefill. Normal automatic
expert placement and adaptation remained enabled. Prompt and conversation caches
were disabled. All tested engines loaded without SSD expert spill.

The control is `--spec 8`, `--mtp-max-t 0`, `--suffix-draft 0`.
A zero MTP cap follows `spec`; `spec=1` is the no-drafting baseline.

## Measurement

Two deterministic counting prompts generate 2,048 tokens each. Combined decode
throughput counts both slots only from the second stream's first output until
the first stream's last output. The other scenario starts an 8,192-token counting
reply, then submits an uncached 21,306-token garden prompt. Decode throughput is
measured only from submission of that prompt until its first output. The existing
decoder remained active through the full prefill in every run.

Actual per-slot generated-token counters are sampled every 20 ms; content-event
counts are not used as token counts. Interval boundaries therefore have sampling
uncertainty. Streaming gaps use client-observed nonempty content events, which
can contain more than one token. They are not exact engine inter-token gaps.

The initial order was 8, 4, 1, 2, 6, dual decode then overlap. The strongest
non-control dual-decode candidate (6) and control (8) were repeated in that order,
with overlap before dual decode. Cache placement was not pinned and adaptation
history differs by scenario order; these are deployed-setting comparisons,
not isolated arithmetic timings. Counting prompts favor predictable drafting.

## Results

Ranges show two passes for 6 and 8; other settings have one pass.
Power limits for every row are 400 / 320 / 250 W.

| spec | Combined decode tok/s | Decode during prefill tok/s | Prefill seconds | Dual worst-slot p95 event gap ms | Dual largest event gap ms |
|---|---:|---:|---:|---:|---:|
| 1 | 111.5 | 62.0 | 46.65 | 38.7 | 72 |
| 2 | 106.8 | 101.1 | 40.83 | 40.5 | 288 |
| 4 | 159.3 | 157.8 | 31.62 | 33.0 | 293 |
| 6 | 182.7–219.8 | 166.1–196.8 | 32.36–40.43 | 31.7–44.4 | 771 |
| 8 | 199.6–220.5 | 191.0–231.5 | 34.24–41.42 | 42.5–50.6 | 1234 |

At 8 versus 6, mean combined decode was 210.1 versus 201.2 tok/s; mean decode
during prefill was 211.2 versus 181.5 tok/s. Mean prefill time was 37.83 versus
36.40 s. During prefill, the largest observed event gap was 487 ms at 8 and
439 ms at 6. No hard streaming-gap threshold was specified.

Keep 8 for the throughput preference, with the measured pause tradeoff above.
The dual-decode repeat included a 1.23 s pause at 8, so this is not a claim of
uniform latency. More varied prose/tool workloads and more repeats would be
needed to establish a general optimum. Suffix drafting was not swept. No engine
code or production MTP setting changed; no fresh arithmetic quality gate is
implied by this configuration sweep.
