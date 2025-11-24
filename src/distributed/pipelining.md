# Pipelining

Pipelining is a model parallelism strategy used when a model is too large to fit into the memory of a single device. Instead of sharding the data, we partition the model itself.

We vertically slice the model and assign a group of layers to each device. For example, if we have 4 devices and 16 layers: Device 0 holds layers 0-3, Device 1 holds layers 4-7, and so on.

## Inference

During inference, Pipeline Parallelism acts like a factory assembly line.

We can achieve 100% utilization by keeping the pipeline full. As soon as Device 0 finishes processing Request A and passes it to Device 1, it immediately picks up Request B. We fully overlap communication with computation.

## Training

Training is significantly harder due to the dependency between the forward and backward pass.

The Dependency: We cannot perform the backward pass for a batch until the forward pass is complete.

This creates a wait time known as the Pipeline Bubble. Device 3 sits idle waiting for the first data to arrive, and Device 0 sits idle waiting for the last gradients to return (cool-down).

**Solution: Micro-batching** To minimize the bubble, we split the global batch into smaller micro-batches. By processing smaller chunks, we can pass data to the next device sooner, allowing the pipeline to fill up faster.

## Pros and cons

Pipelining is particularly useful during inference, it maximizes throughput by making sure all devices are active at all times. However, the  latency of an individual request is higher than with other strategies because it does not use all the chips in parallel for a single prompt but sequentially. It is mostly suitable for high throughput, non latency constrained scenarios.

| Feature | Impact | Why? |
| :--- | :--- | :--- |
| **Throughput** | ✅ **High** | During inference (or well-tuned training), all devices work in parallel. |
| **Communication** | ✅ **Low** | We only send activations between devices at the boundaries (e.g., after layer 4). This is much cheaper than Tensor Parallelism. |
| **Latency** | ❌ **High** | A single request must travel **sequentially** through all devices. Device 4 cannot start until Device 3 finishes. |
| **Memory** | ✅ **Efficient** | Parameters are **split across devices**, allowing us to fit larger models. |