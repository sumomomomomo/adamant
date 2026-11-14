# DFlash2

[简体中文](zh/dflash2.md)

`--dflash2 PACK` selects the official Flash-Next DFlash2 block drafter instead of MTP.
The target model, sampler and grammar verifier remain responsible for every emitted token.
MTP remains the default. This is an opt-in CUDA port of the implementation in
[Strata PR 3322](https://github.com/Niko1221/Strata/pull/3322), revision
`c90fa866c36c5e4c10ca7111a1e4f2af311df49e` (MIT).

## Prepare the drafter

Download `model.safetensors` and `config.json` from
[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, then run:

```sh
python tools/convert/dflash2_pack.py model.safetensors config.json dflash2.df2p
bin/basalt serve --model target.basalt --dflash2 dflash2.df2p \
  --spec 4 --prompt-cache 4 --prefill auto --resident-experts
```

Conversion needs NumPy. The runtime pack contains Q8_0 projection matrices,
F32 normalization/convolution vectors and BF16 selector codebooks. It shares the
complete target embedding and output head; it does not restrict the draft vocabulary.
The runtime understands Adamant's planar Q8_0 output-head layout.

## Execution

The drafter reads the raw mean of the target's HC streams after layers
5, 15, 25, 35 and 45. Capture kernels only read target buffers.
An eight-row block contains the anchor and seven masked positions. A
predecessor-conditioned top-16 selector proposes seven tokens; `--spec` limits
the verification window, including its anchor. Prompt lookup may also supply proposals.

Only committed target rows enter the private per-layer KV rings. Rejected
proposals never enter these rings. Fresh conversations reset the draft context;
continuations restore it alongside the target state.
Attention sees the checkpoint's 2,048-token local window, with each query's own
position mask and all eight block positions. Prompt capture keeps at most the
last 2,048 rows of a chunk. Its memory does not grow with prompt length.

The drafter's weights, rings, prompt capture and scratch are allocated before
expert-cache sizing. Target weights and their precision are unchanged. Draft
Q8_0 arithmetic can change proposal acceptance; proposals still require target
verification. See [precision](precision/dflash2.md).

## Supported serving configurations

Single-slot serving supports one GPU or multiple GPUs used as expert tiers.
With a packed vision projector and `--vision dynamic`, DFlash2 also drafts after
image prompts and on later turns containing image history. Target vision keeps
its existing M-RoPE positions; draft attention uses contiguous logical row
ordinals. Image rows supply target HC features just like text rows.
Batch slots and splitting target layers across GPUs are rejected.
The existing MTP serving path retains its defaults.

## Conversation caching

Prompt checkpoints, RAM parking and disk parking include the private draft KV
state. For example, append:

```sh
--prompt-cache 4 --conversation-cache-mib 2048 \
  --conversation-cache-slots 4 --conversation-cache-min-free-mib 6144 \
  --conversation-cache-disk-mib 8192 --conversation-cache-disk-min-tokens 0 \
  --conversation-cache-dir /path/to/conversation-cache
```

`--prompt-cache` bounds retained checkpoints; `--conversation-cache-slots`
bounds parked RAM conversations. MiB budgets bound each tier. The disk minimum
is 8,192 tokens by default; zero allows short conversations too. Disk entries
survive server restarts and are restored only for a matching model/configuration
and drafter identity. Image content is part of prefix matching.

A draft snapshot stores the last `min(position, 2047)` committed F32 K/V rows
and a 64-byte header. Its maximum size is 83,845,184 bytes (about 79.96 MiB)
per checkpoint, in addition to target state. The payload is copied without
quantization. Restore validates the drafter identity, position, geometry and
length before writing GPU state. Disk format version 3 carries this extra state;
older cache files are not reused.

## Validation

```sh
python tools/test_dflash2_pack.py
cmake --build build --target dflash2_kernels_parity dflash2_parity
build/dflash2_kernels_parity
build/dflash2_parity dflash2.df2p reference-case.bin
```

The kernel test compares real model geometries with CPU double references,
checks the HC tap against the existing gated write, and prints kernel timings.
The artifact test consumes a synthetic reference case with final block rows,
selector tokens and probabilities, including context longer than the local window.
It also overwrites and restores short and ring-wrapped draft snapshots and
requires bit-for-bit replay. CPU conversation-store tests cover the extra state
through disk round trips and corruption checks.
