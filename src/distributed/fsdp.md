# Fully Sharded Data Parallel (FSDP)

With FSDP, not only do we shard the batch over multiple chips like with [Data Parallelism](./data.md), we also shard the optimizer state, the gradients and the paramerters over multiple chips, thus enabling us to train models that would not fit on a single chip.

## Gather-Compute-Discard

The main mechanism behind FSDP is called **Gather-Compute-Discard**.

0. **Shard:** We initially fully shard tensors to reduce the per-chip memory.
1. **All Gather (Weights):** Before the forward pass of a layer, we [All Gather](./all_gather.md) the parameters so that each chip momentarily holds a full replica of that specific layer.
2. **Compute:** We compute the forward/backward pass with the full layer.
3. **Discard (Weights):** We delete the parts of the tensor our chip did not initially owned to reduce memory requirements.
4. **Reduce-Scatter (Gradients):** After the backward pass, instead of [All Reducing](./all_reduce.md) (which keeps a full copy of gradients everywhere), we Reduce Scatter the gradients. Each chip ends up with only the specific chunk of gradients corresponding to the parameters it owns.

*Note: The communication is typically overlapped with computation to hide latency.*

## The three stages

Sharding more tensors means increasing the amount of All Gather'ed data. Ideally we would shard as little as possible. Nonetheless, if only sharding the optimizer's state is not enough, we need to shard the gradients as well, or even the model parameters.

We often refer to the levels of sharding as the **Three Stages of ZeRO**, after [Deepspeed's ZeRO paper](https://arxiv.org/abs/1910.02054).

| Stage | What is Sharded? | Memory Savings | Communication Overhead |
| :--- | :--- | :--- | :--- |
| **ZeRO-1** | **Optimizer States only** | **~4x reduction.** (Optimizer states are typically 75% of training memory). | **Minimal** (Same as DDP). |
| **ZeRO-2** | **Optimizer + Gradients** | **8x reduction.** | **Minimal.** |
| **ZeRO-3** | **Opt + Grads + Parameters** | **Linear reduction** ($1/N$). Allows fitting massive models. | **High.** Requires All-Gather before every layer. |

## Why is the optimizer state so large?

For every parameter, we hold:

- 2 bytes (bf16 weight)
- 2 bytes (bf16 gradient)
- 12 bytes (f32 optimizer state: master copy, momentum, variance) Total: 16 bytes. By sharding just the optimizer states (12 bytes), we remove 75% of the memory footprint with almost zero communication penalty.

## Pros and Cons

FSDP should only be used during training. It saves memory but doesn't speed up the math for a single sample; in fact, the communication overhead would make generation slower.
