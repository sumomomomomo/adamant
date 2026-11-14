# DFlash2 precision

[简体中文](../zh/precision/dflash2.md)

[Serving and conversion](../dflash2.md)

DFlash2 only proposes tokens. The existing target verifier, penalties, grammar
masking and sampler decide every emitted token. The port adds read-only target
feature capture and does not change target weights, arithmetic or state updates.
Comparisons must pin actual expert placement: changing the drafter changes free
VRAM, and automatic placement can change expert reduction order.

Image requests retain the existing target vision/M-RoPE handling. DFlash2 captures
image-row HC features and indexes its private attention by logical row ordinal.
These coordinate systems remain separate, including after cached restores.

## Draft arithmetic

The runtime pack uses Q8_0 projection matrices with planar int8 activations and
F32 activation scales. Normalization, dynamic convolution, rotary positions and
attention accumulate in F32. The predecessor/successor codebooks stay BF16.
Conversion rounds Q8_0 codes to nearest, ties to even. The full target embedding
and head are shared in their existing formats.

The private KV ring holds F32 keys and values. HC means are captured from the
completed target layer output. The deferred FFN-write capture computes its mean
without modifying the residual; regrouping additions can differ slightly from
performing each stream's write and then averaging. This affects proposals only.

## Development validation

CUDA kernel parity passed on sm_86: RTX 3090 Ti, RTX 3090 and RTX 3080.
All tested operations have CPU double references at the real model geometry;
quantized outputs allow half one quantization step plus floating-point error.
The HC capture additionally compares with the existing gated-write kernel.

On the RTX 3090 Ti, capture error against the existing write was at most
`4.77e-7` absolute. The Q8 GEMM's maximum absolute error against a double dot was
`6.72e-5` in the tested matrices. Near-zero reference values can have large
relative error; the test reports both absolute and relative maxima.

An independent FP32 PyTorch reference using the original BF16 draft weights
was checked on a random 2,300-token context and 4,096-row synthetic head.
Final-row cosine similarity ranged from `0.98791` to `0.99671`; six of seven
selector tokens matched. This comparison includes draft weight and activation
quantization. It is not a target-model quality measurement.

Kernel timings are development measurements at power limits of
375 W / 320 W / 250 W for the three cards respectively. The real-geometry
attention kernel measured about 55 / 65 / 70 microseconds per call, and the
3090 Ti's largest tested fusion projection reached about 735 GB/s.
These are microbenchmarks, not end-to-end speed claims.

With a fixed 16.0 GiB primary expert tier, one GPU and adaptation disabled,
MTP and DFlash2 produced identical greedy output for all 384 checked tokens.
This isolates target arithmetic from automatic placement changes.

The adversarial JSON-schema check passed through DFlash2 serving, including
masked resampling after rejected proposals. Validation commands are on the
[serving page](../dflash2.md).

## Cache-state exactness

Draft snapshots copy F32 K/V bytes without cache quantization and retain only
the last 2,047 committed rows that the next query can attend to. Restore checks
an identity derived from the complete draft weight contents, logical position,
geometry and payload size before transfer. These operations leave target
arithmetic unchanged.

At 7-row and 2,300-row contexts, overwriting the entire private ring and restoring
it reproduced draft tokens, probabilities and final-row floats bit for bit.
Serving checks cover image follow-up turns, histories longer than the local
window, RAM conversation switching and disk restores across a restart. Greedy
answers match between cold and cached runs; changing an image only reuses the
unchanged prefix.
