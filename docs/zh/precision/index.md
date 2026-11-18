# 精度

[English](../../precision/index.md)

**Adamant：** 本移植的执行路径及其验证状态见 [Ada 的执行与验证](ada.md)，其中也列出 Ampere（sm_86）上未完成的工作。可选的 [DFlash2 草稿模型](dflash2.md)有独立的精度与验证记录。
实验性[大分块对等预填充](large-peer-prefill.md)在保留专家运算的前提下测试双设备调度。
[短提示词调查](small-prefill.md)记录传输成本、调度实验及额外专家辅助卡原型。
[额外专家层级路径](extra-peer-prefill.md)记录其受限派发、内存回退及 BF16 质量比较。
[诊断性长提示词预填充实验](long-prefill.md)记录 Q5_0 融合下投影、IQ4_XS v2 门控/上投影及大分块验证。

[实验性三 GPU 阈值](peer-prefill-cutoffs.md)记录路由与流式路径的切换点及质量检查。
[双卡长提示词流水线](paired-prefill.md)说明按完整层调度、匹配 HC8 权重和检查点恢复。

Adamant 自己的测量结果在
[README](../../../README.zh-CN.md#ada-上的实测)。下面的页面描述的是 Blackwell 上的上游 Basalt；其中的测量环境与数字
不是 Adamant 的结果。

每个模型一页，说明它以什么精度计算、每种舍入要付出多少代价。方法是共用的。

- [Qwen3.8-Flash-Next](qwen38-flash-next.md)：各阶段、prefill、实测成本、起草与有损标志。
- [质量如何测量](quality.md)：门槛、BF16 参考基准、贪心一致性与复现命令。
