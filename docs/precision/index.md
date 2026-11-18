# Precision

[简体中文](../zh/precision/index.md)

**Adamant:** the Ada port's paths and their validation state are in [Ada execution and validation](ada.md),
which also lists the open Ampere (sm_86) work.
The optional [DFlash2 drafter](dflash2.md) has its own precision and validation record.
The experimental [large-chunk peer prefill](large-peer-prefill.md) preserves expert arithmetic while testing two-device scheduling.
The [small-prompt investigation](small-prefill.md) records transfer costs, scheduling experiments and an additional expert-peer prototype.
The [additional expert-tier path](extra-peer-prefill.md) records its bounded dispatch, memory fallback and BF16 quality comparison.
The diagnostic [long-prefill experiment](long-prefill.md) records Q5_0 fused down projections, IQ4_XS v2 gate/up, and large-chunk validation.

The [experimental three-GPU cutoffs](peer-prefill-cutoffs.md) record the routed/streamed crossover and its quality checks.
The [paired-card long-prefill pipeline](paired-prefill.md) covers whole-layer scheduling, matching HC8 weights and checkpoint recovery.

Adamant's own measurements are in the [README](../../README.md#measured-on-ada). The pages below describe upstream
Basalt on Blackwell; their setup and numbers are not Adamant results.

Each model has one page for what it computes in and what each rounding costs. The method is shared.

- [Qwen3.8-Flash-Next](qwen38-flash-next.md): stages, prefill, the measured costs, drafting and the lossy flags.
- [How quality is measured](quality.md): the gate, the BF16 reference, greedy identity and reproduction commands.
