# All-Reduce And Reduce-Scatter

An All-Reduce operation takes a tensor from every device, combines them using an operator (typically `sum`), and returns the **full result** to every device.

For instance, a vector of length `256` whose single axis would be sharded over 4 devices:

![img](./all_reduce.png)

Each TPU initially holds 64 unique elements, after the `All-Reduce`, they all hold a vector which is replica of the sum of the vectors initially held by each chip.

## Reduce-Scatter

A Reduce-Scatter is a "fused" operation. It performs the same reduction (sum) as All-Reduce, but instead of returning the full result to everyone, it scatters (shards) the result across the devices.

Conceptually, `Reduce-Scatter` is equivalent to an `All-Reduce` followed immediately by a Scatter (slice), but it is much more bandwidth-efficient because the full sum is never fully materialized on any single device.
