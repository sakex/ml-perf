# Tensor Parallelism (TP)

Tensor Parallelism shards the model's parameters and synchronizes the activations using collective operations. It allows saving on memory per chip while also lowering latency by splitting computations across multiple devices.

**Unlike FSDP (which shards weights but gathers them before compute), TP shards the computation itself.**

There are multiple ways to implement Tensor Parallelism. The best method will depend on your model's architecture and dimensions. A common example for LLMs is [Megatron Sharding](https://arxiv.org/abs/1909.08053).

## How to think about TP?

A good rule of thumb is that you want to minimize the amount of data being transfered between devices because the interchip connection is much slower than the local memory.

You also have to understand which operations will require synchronization to remain correct. Basically, whenever a sharded axis gets reduced, the output needs to either be [All-Reduced](./all_reduce.md) or the input needs to be [All-Gathered](./all_gather.md) beforehand. Other operations do not need to be synchronized.

Let's chain two dot products and evaluate how we can shard:

```python
y = torch.einsum('btd,df->btf', x, w0)
y = torch.relu(y)
mlp_out = torch.einsum('btf,fd->btd', y, w1)
out = mlp_out + x
```

`b` is batch, `t` is sequence, `d` is model dimension, `f` is hidden dimension.

- Sharding on `b` or `t` is simple data parallelism
  - No collective operations will be needed.
- Sharding on `d`
  - `d` is only reduced during the first einsum, therefore we only need to synchronize around the first einsum.
  - If we only shard either the activations or the weights but not both, we can [All-Gather](./all_gather.md) the sharded tensor before executing the first einsum.
  - If we shard both, we can [All-Gather](./all_gather.md) both. But more efficiently, we can perform the first einsum with the local data and then [All-Reduce](./all_reduce.md) the output.
  - The final addition never needs any synchronization because devices share the same indices for both arrays and do not reduce the axis.
- Sharding on `f`
  - `f` is only reduced during the second einsum.
  - If `w0` is sharded on `f`, the output of the first einsum will also be sharded on `f`.
  - If both `w0` and `w1` are sharded on `f`, we need to [All-Reduce](./all_reduce.md) the output of the second einsum before performing the final addition (or after, the order does not matter.)
  - If only one of the weights is sharded on `f`, we can [All-Gather](./all_gather.md) the sharded tensor.
- Sharding on both `d` and `f`, and even `b`, `t`
  - We can shard on any combination of the axes, the same rules will apply.
