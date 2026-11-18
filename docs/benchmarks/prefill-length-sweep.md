# Prefill throughput against prompt length

[简体中文](../zh/benchmarks/prefill-length-sweep.md)

Development sweep recorded 2026-11-18 on RTX 3090 Ti / RTX 3090 / RTX 3080 at
enforced 400 / 320 / 250 W limits, with 62 GiB system RAM. It measures prefill
throughput over a ladder of fresh prompt lengths on the deployed serving
configuration. These are agent-run development measurements, not maintainer
headline results, and they are not a quality comparison.

## Result

The engine ran from the 32K-boundary build (git `f259374`, binary SHA-256
`bdcadb49b6530084d89b82d8cfe9c261a5502f34807d9ea5c995d59265176818`) with
automatic expert and prefill placement, adaptation and disk parking active.
Settings matched the production unit exactly apart from the listening host and
port: `--gpu 0,1,2`, int8 KV, context 262,144, `--prefill 32768`, DFlash2 at
spec 8, prompt cache 4, disk conversation cache enabled. The expert cache
settled to 7,543 slots. Each point is one fresh request with zero prompt-token
reuse, three repetitions in rotated order, from a single engine start.
`prompt_n` was confirmed equal to the nominal length on every request. Rates are
the engine's own prompt token counts over engine prompt time.

| Fresh tokens | Median tok/s | Range tok/s | Best tok/s | Median ms/token | Prompt path |
|---:|---:|---:|---:|---:|:--|
| 256 | 412 | 372–482 | 482 | 2.425 | primary only |
| 512 | 740 | 612–913 | 913 | 1.351 | primary only |
| 1,024 | 962 | 949–1,243 | 1,243 | 1.039 | primary only |
| 2,048 | 1,271 | 1,064–1,470 | 1,470 | 0.787 | primary only |
| 4,096 | 1,532 | 1,525–1,572 | 1,572 | 0.653 | primary only |
| 4,097 | 1,519 | 1,473–1,565 | 1,565 | 0.658 | primary only |
| 6,144 | 1,617 | 1,173–1,623 | 1,623 | 0.618 | primary only |
| 6,145 | 1,622 | 1,617–1,660 | 1,660 | 0.617 | primary only |
| 8,192 | 2,008 | 1,981–2,018 | 2,018 | 0.498 | primary only |
| 10,240 | 2,305 | 1,659–2,436 | 2,436 | 0.434 | primary only |
| 12,288 | 2,831 | 2,736–2,866 | 2,866 | 0.353 | primary only |
| 14,336 | 3,042 | 2,552–3,179 | 3,179 | 0.329 | primary only |
| 16,384 | 3,399 | 3,153–3,488 | 3,488 | 0.294 | primary only |
| 24,576 | 2,822 | 2,793–4,167 | 4,167 | 0.354 | primary only |
| 32,768 | 4,264 | 3,707–4,344 | 4,344 | 0.235 | primary only |
| 32,769 | 4,140 | 4,117–4,336 | 4,336 | 0.242 | primary only |
| 49,152 | 4,297 | 4,095–5,022 | 5,022 | 0.233 | 1 chunk of 24,576 on the first peer |
| 65,536 | 4,105 | 3,914–4,705 | 4,705 | 0.244 | 1 chunk of 32,768 on the first peer |
| 98,304 | 5,306 | 4,859–5,603 | 5,603 | 0.188 | 2 chunks of 32,768 on the first peer |
| 131,072 | 5,380 | 5,316–6,045 | 6,045 | 0.186 | 3 chunks of 32,768 on the first peer |
| 163,840 | 5,537 | 5,203–6,048 | 6,048 | 0.181 | 4 chunks of 32,768 on the first peer |
| 196,608 | 5,477 | 5,456–5,501 | 5,501 | 0.183 | 5 chunks of 32,768 on the first peer |
| 229,376 | 5,559 | 5,525–5,586 | 5,586 | 0.180 | 6 chunks of 32,768 on the first peer |
| 245,760 | 5,898 | 5,543–5,993 | 5,993 | 0.170 | 7 chunks of 30,720 on the first peer |

Below about 8,192 tokens the cost per token is dominated by per-request fixed
cost: it falls 14-fold, from 2.425 to 0.170 ms/token, as that fixed cost is
amortized rather than because the cards speed up. Between 32,768 and 65,536
tokens the rate flattens near 4,100–4,300 tok/s and does not improve with
length. From 98,304 tokens the rate saturates near 5,500 tok/s; the last five
points span 0.170–0.186 ms/token, and the two longest, 196,608 and 229,376
tokens, repeat within about 1%. That is consistent with the measured 13.3 GB/s
host-to-device limit rather than with compute. Peer chunk count grows almost
linearly with length and is a useful secondary series when plotting.

## The measurement confound: expert-cache reuse between requests

A 24,576-token request first appeared to be a regression: its median rate
(2,822 tok/s) was below both its neighbours, and in absolute time it was slower
than the 32,768-token request that carries 33% more tokens. Per-token cost
cannot rise with length, so the point could not be a context-length effect.

It was not. The three repetitions are bimodal:

| Repetition | Prompt time | Rate | Routed experts streamed over PCIe | Preceding request |
|---|---:|---:|---:|---|
| 0 | 8,799 ms | 2,793 tok/s | 45.2% | 16,384 tokens |
| 1 | 5,898 ms | 4,167 tok/s | 28.5% | 32,768 tokens |
| 2 | 8,709 ms | 2,822 tok/s | 48.3% | 229,376 tokens |

In its fast mode 24,576 tokens sits exactly on the length curve
(0.2400 ms/token, between the 0.2870 of 16,384 tokens and the 0.2300 of 32,768
tokens), so there is no knee. The median had simply selected the contaminated
mode, two slow samples out of three.

The cause is that expert-cache VRAM is the elastic donor for KV backing. A long
request reclaims those slots for its own KV and evicts the experts held there;
the engine logs this as `elastic KV: … the expert cache N -> M slots, K experts
displaced`. The following request then streams the experts its prompt routes to
over PCIe instead of finding them resident. In the slow repetition above, the
preceding 229,376-token request had taken 245,760 KV cells and displaced 1,251
experts, leaving 6,243 slots; the fast repetition began with 7,398 slots intact.

The discriminator across all 75 requests is the share of routed experts read over
PCIe, reported as `… more read by the GPU over PCIe or from another GPU (P% of
all 480 routed)`. It correlates with the shortfall against the best rate observed
at the same length at r = −0.354 (n = 50, excluding the cold first segment), and
per-request SSD expert reads are monotone in it: requests reading 0–30 GB from
the file average a 4.8% shortfall, those reading 80–200 GB average 14.5%. The
entry slot count is a weaker predictor (r = +0.14), because what matters is
whether the specific experts a prompt routes to are resident, not how many slots
are free; the segment that ran cold shows this clearly. 13 of the 72 timed
requests ran more than 15% below the best rate ever measured at their own
length, across 10 different lengths.

Consequences for sweeps of this kind:

- **Report best-of-N, not median-of-N.** A median understates throughput at any
  length whose cell is bimodal, and hides the effect entirely.
- **Prime each point.** A same-length warm-up request with a different unique
  prefix leaves the expert working set resident while still reporting zero
  prompt-token reuse. Without it, each timed request's cost depends on whichever
  point the rotation happened to place before it.
- **Long points are self-priming.** At 196,608 tokens and above the request warms
  its own working set internally and the confound washes out; the short and
  middle of the ladder need the priming most.
- **A unique prefix keeps the prompt cache honest, and nothing else.** Fresh
  prefixes per request are what enforce the zero-reuse assertion; they do not
  control expert residency.

## Secondary observation

In this run the paired peer path was not selected below 49,152 fresh tokens, and
32,769 tokens measured no better than 32,768 (4,140 against 4,264 tok/s). The
committed [paired-card long prefill](../precision/paired-prefill.md) page records
eligibility above 32,768 fresh tokens per prompt part, and other runs on this
machine with a pinned expert cache do split at 32,769. The cost model decides, and
with placement, adaptation and disk parking left automatic here, the observed
onset was higher. This is an observation from one run under production settings,
not a correction to that page's measurements.
