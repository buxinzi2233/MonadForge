# V100 16 GB：Anima T-LoRA 无 compile 训练实验交接记录

记录日期：2026-07-25

## 仓库状态

- 仓库：`/home/buxinzi/Projects/toolbox/model-train/MonadForge`
- Fork：`https://github.com/buxinzi2233/MonadForge`
- 分支：`experiment/eager-lora-down-autograd`
- 本记录创建前的代码提交：`c79ff0a9 perf: enlarge eager V100 training chunks`
- 远端跟踪分支：`fork/experiment/eager-lora-down-autograd`

相关提交：

```text
c79ff0a9 perf: enlarge eager V100 training chunks
57b2604b perf: reduce eager V100 LoRA kernel fragmentation
fb591f46 feat: bound eager V100 LoRA training memory
09fb27e6 feat: reduce eager fp32 LoRA activation memory
93495fd5 backup: snapshot before eager LoRA memory experiment
```

## 目标配置

当前实验解决的是 V100 16 GB 上如下 Anima T-LoRA 训练：

```toml
resolution = 1024
gradient_checkpointing = false
blocks_to_swap = 0
torch_compile = false
gradient_accumulation_steps = 4
mixed_precision = "fp16"
attn_mode = "mem_efficient"
```

这里的精度策略不能描述成“把 FP16 切换到 FP32”。实际策略是为避免
FP16 训练溢出和 NaN 而设计的混合精度：

- 冻结主干的子层 GEMM 仍使用 FP16。
- LoRA rank GEMM 使用 FP32。
- gated residual 累加使用 FP32。
- 保留 FP32 residual，避免关键残差路径在 FP16 下溢出或溢出。

## 已确认的 compile 显存机制

`torch_compile=true` 比 eager 少约 2 GiB 显存，并不是因为
`torch.compile` 本身天然节省显存，而是当前配置同时启用了：

```text
library/runtime/harness.py::_apply_activation_memory_budget
activation_memory_budget = 0.99
```

它使 AOTAutograd min-cut partitioner 对联合前后向图执行选择性重计算。
因此当前 compile 路径同时获得了：

1. 编译和融合带来的速度收益；
2. 全局 saved-for-backward 激活规划带来的显存收益。

实测：

| 路径 | peak allocated | peak reserved |
|---|---:|---:|
| eager 自定义路径 | 14.217 GiB | 14.496 GiB |
| compile + budget 0.99 | 12.152 GiB | 12.453 GiB |

两条路径进入 forward 前都约为 4.09 GiB。因此差异来自 forward 保留给
backward 的激活，不是 CUDA allocator cache。

关键隔离实验：

```toml
torch_compile = true
activation_memory_budget = 1.0
```

冷缓存首次 forward OOM：

```text
15.29 GiB allocated
47.99 MiB reserved but unallocated
Tried to allocate 18 MiB
```

结论：`activation_memory_budget=0.99` 是 compile 配置能装入 16 GB 的
关键条件，相比 budget 1.0 至少减少约 3.14 GiB。eager 自定义 autograd
只能做局部 chunk/recompute，无法复制 AOTAutograd 的全局分区规划。

形状量级参考：

```text
sequence ≈ 4200
width = 2048
blocks = 28
28 个 block 的一组 FP32 全宽激活 ≈ 0.897 GiB
```

eager 和 compile 的 2.065 GiB 差值约等于 2.3 组这种激活。

相关背景文档：

```text
docs/findings/custom_autograd_removal_partitioner_oom.md
```

## 当前 eager 实现

为了在无 gradient checkpointing、无块交换、无 compile 的条件下装入
V100 16 GB，当前分支增加了 eager 专用的局部重计算和分块自定义
autograd 路径。主要代码：

```text
library/anima/eager_autograd.py
networks/lora_modules/custom_autograd.py
library/anima/models.py
networks/lora_modules/base.py
networks/lora_modules/lora.py
train.py
```

当前经过实机调优的分块常量：

```python
# library/anima/eager_autograd.py
_EAGER_ROPE_SEQ_CHUNK = 8192
_EAGER_MLP_ROW_CHUNK = 3072

# networks/lora_modules/custom_autograd.py
EAGER_LORA_CHUNK_ROWS = 3072
```

调整前：

```python
_EAGER_ROPE_SEQ_CHUNK = 256
_EAGER_MLP_ROW_CHUNK = 1024
EAGER_LORA_CHUNK_ROWS = 2048
```

扩大 chunk 的目的不是继续降低显存，而是减少 eager Python/autograd
分块数量、局部图重建次数和小 kernel 启动开销。

## 性能结果

局部 V100 microbenchmark：

### MLP，`4200×2048 -> 8192 -> 2048`

```text
chunk 1024: ~19.18 ms
chunk 3072: ~16.83 ms
```

单个 fused MLP 约快 12%。

### qkv LoRA，`2048 -> 6144`，rank 32

```text
chunk 2048: ~4.98 ms
chunk 3072: ~4.48 ms
```

代表性 LoRA projection 约快 10%。

### 完整训练 A/B

两组独立六步训练结果一致：

| 版本 | 速度 | peak allocated | peak reserved |
|---|---:|---:|---:|
| 调整前 | 7.09–7.11 s/optimizer step | ~14.39 GiB | ~14.54 GiB |
| 当前版本 | 6.56–6.57 s/optimizer step | ~14.59 GiB | ~14.70 GiB |

结果：

- 完整训练约提速 7.7%。
- 代价是 peak allocated 增加约 200 MiB。
- 六个 optimizer step 内无 OOM、无 NaN。
- 用户先前手工观察约为 6.8–7.0 s/step；不同 bucket 下波动是正常的，
  当前代码的预期实用区间约为 6.5–6.7 s/step。

临时日志仍在本机：

```text
/tmp/monadforge-eager-speed-baseline.log
/tmp/monadforge-eager-speed-optimized.log
/tmp/monadforge-eager-speed-baseline-seed123.log
/tmp/monadforge-eager-speed-optimized-seed123.log
```

临时 microbenchmark：

```text
/tmp/bench_eager_mlp_chunks.py
/tmp/bench_eager_lora_chunks.py
/tmp/bench_eager_rope_chunks.py
```

注意：名称带 `seed123` 的临时 TOML 可能把 `seed=123` 追加到了 TOML
array table 后，而不是顶层。不过两组独立 A/B 的速度差几乎相同，因此
约 7.7% 的差值仍可信。

## 已通过测试

```text
tests/test_anima_eager_autograd.py       4 passed
tests/test_lora_eager_autograd.py       11 passed
tests/test_mixed_precision_resolver.py
tests/test_network_cfg.py               40 passed combined
```

合计：

```text
55 passed
```

## 当前判断

`~6.56 s/step` 已接近低风险、纯 eager、仅调整 chunk 的实用上限。

当前主要瓶颈：

1. `EagerFusedLoRAMLPFn.backward` 每个 chunk 都重建局部计算图并调用
   `torch.autograd.grad(...)`。
2. 关闭 compile 后没有全局 kernel fusion 和全图显存调度。
3. memory-efficient attention 和冻结主干 GEMM 已较高效，并占据较大
   比例。
4. V100 上 FlashAttention 对当前 Anima FP16 训练不够稳定，既往实测
   self-attention 出现 NaN，不应作为生产方案。
5. 继续扩大 chunk 收益很小，并会提高 OOM 风险。
6. 手工 CUDA Graph 不适合当前动态 token bucket、T-LoRA mask 变化、
   gradient accumulation 和紧张显存条件；graph pool 也会增加显存。

## 下一步可做的高风险优化

如果继续优化，优先目标是：

```text
library/anima/eager_autograd.py::EagerFusedLoRAMLPFn
```

用解析 backward 替换其中嵌套的 `torch.autograd.grad`，显式实现：

- frozen base linear 的 input gradient；
- GELU backward；
- FP32 LoRA down/up 参数梯度；
- FP32 rank gradient 累加；
- 与当前混合精度语义一致的 cast 顺序；
- T-LoRA mask 和 channel scaling。

预计完整训练可能再提升约 3–8%，但这个估计尚未经过真实训练验证。

必须执行的验证：

1. 单元测试逐项比较当前实现和解析 backward 的 output 及全部 gradient。
2. 覆盖 channel scaling 开启和关闭。
3. 覆盖非平凡 T-LoRA mask。
4. V100 上执行 6–10 个 optimizer step 的真实 A/B，记录速度、peak
   memory 和 finite check。
5. 保留现有混合精度策略，不能把整个主干改为 FP32。
6. 只有确认实测提速且稳定后再提交和推送。

## 新对话接续建议

新对话应先检查：

```bash
cd /home/buxinzi/Projects/toolbox/model-train/MonadForge
git status --short --branch
git log -5 --oneline
```

然后从本文件和以下文件开始：

```text
docs/findings/eager_v100_no_compile_handoff_20260725.md
docs/findings/custom_autograd_removal_partitioner_oom.md
library/anima/eager_autograd.py
tests/test_anima_eager_autograd.py
```
