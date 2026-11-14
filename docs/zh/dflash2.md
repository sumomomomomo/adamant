# DFlash2

[English](../dflash2.md)

`--dflash2 PACK` 使用官方 Flash-Next DFlash2 分块草稿模型替代 MTP。
目标模型、采样器和语法验证器负责验证每个输出 token。
默认仍使用 MTP。这是对 [Strata PR 3322](https://github.com/Niko1221/Strata/pull/3322)
中 CUDA 实现的可选移植，版本为
`c90fa866c36c5e4c10ca7111a1e4f2af311df49e`（MIT）。

## 准备草稿模型

从 [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
下载 `model.safetensors` 和 `config.json`，版本为
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`，然后运行：

```sh
python tools/convert/dflash2_pack.py model.safetensors config.json dflash2.df2p
bin/basalt serve --model target.basalt --dflash2 dflash2.df2p \
  --spec 4 --prompt-cache 4 --prefill auto --resident-experts
```

转换需要 NumPy。运行时文件包含 Q8_0 投影矩阵、F32 归一化及卷积向量，
以及 BF16 选择器码本。它共享目标模型的完整嵌入表和输出头，不缩减草稿词表。
运行时支持 Adamant 的平面布局 Q8_0 输出头。

## 执行

草稿模型读取目标模型第 5、15、25、35、45 层完成后各 HC 流的原始平均值。
捕获内核只读取目标缓冲区。八行分块包含一个锚点和七个掩码位置。
以前驱 token 为条件的 top-16 选择器提出七个 token；`--spec` 限制包含锚点的
验证窗口宽度。提示词查找也可以提供候选。

只有已提交的目标行进入各层私有 KV 环。被拒绝的候选不会进入这些环。
新会话重置草稿上下文；续接时将其与目标状态一起恢复。
注意力使用检查点定义的 2,048 token 局部窗口，
按每个查询自身的位置遮罩，同时可见全部八个分块位置。
提示词捕获最多保留每个分块最后 2,048 行，内存不会随提示词长度增长。

草稿权重、KV 环、提示词捕获和暂存区都在专家缓存定容前分配。
目标权重和精度保持不变。草稿的 Q8_0 运算可能改变候选接受率，
但所有候选仍须经目标验证。参见[精度说明](precision/dflash2.md)。

## 支持的服务配置

支持单槽服务，可使用单 GPU 或将多个 GPU 用作专家层级。
模型文件包含视觉投影器并使用 `--vision dynamic` 时，DFlash2 也会在图像提示词后
以及包含图像历史的后续轮次中提出候选。目标视觉路径保留现有 M-RoPE 位置；
草稿注意力使用连续的逻辑行序号。图像行与文本行一样提供目标 HC 特征。
批处理槽及跨 GPU 拆分目标层会被拒绝。现有 MTP 服务路径保留默认行为。

## 会话缓存

提示词检查点、RAM 暂存和磁盘暂存都包含私有草稿 KV 状态。例如追加：

```sh
--prompt-cache 4 --conversation-cache-mib 2048 \
  --conversation-cache-slots 4 --conversation-cache-min-free-mib 6144 \
  --conversation-cache-disk-mib 8192 --conversation-cache-disk-min-tokens 0 \
  --conversation-cache-dir /path/to/conversation-cache
```

`--prompt-cache` 限制保留的检查点数量；`--conversation-cache-slots` 限制暂存于 RAM
的会话数量。MiB 预算分别限制各层容量。磁盘最小长度默认为 8,192 token；设为零也允许
短会话。磁盘条目在服务器重启后仍可使用，但只恢复模型、配置和草稿模型身份匹配的状态。
图像内容参与前缀匹配。

草稿快照保存最后 `min(position, 2047)` 个已提交 F32 K/V 行和 64 字节头部。
每个检查点最多为 83,845,184 字节（约 79.96 MiB），另需保存目标状态。
载荷直接复制，不进行量化。恢复会先验证草稿模型身份、位置、尺寸和长度，再写入 GPU 状态。
磁盘格式版本 3 包含此附加状态；不复用旧格式缓存文件。

## 验证

```sh
python tools/test_dflash2_pack.py
cmake --build build --target dflash2_kernels_parity dflash2_parity
build/dflash2_kernels_parity
build/dflash2_parity dflash2.df2p reference-case.bin
```

内核测试以真实模型尺寸对照 CPU 双精度参考，检查 HC 捕获与现有门控写入的一致性，
并打印内核计时。模型文件测试读取合成参考用例，包含最终分块行、选择器 token 和概率，
并覆盖超过局部窗口长度的上下文。
它还会覆盖并恢复短上下文及环回绕后的草稿快照，要求逐位一致地重放。
CPU 会话存储测试涵盖附加状态的磁盘往返及损坏检查。
