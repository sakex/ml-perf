# Interview Questions

## Does my runtime make sense?

You will be shown some code. You will be asked whether the runtime of the code makes sense given the performance characteristics of a chip. Compute the expected runtime and compare it to the measured runtime.

If the observed runtime is an order of magnitude slower than expected, look for inefficiencies.

- Are the kernels compiled and fused?
- Is the data properly allocated on the GPU or is it copied to and from the CPU?
- Are we using some non vectorized operations that are not being properly fused?
- Are we materializing some intermediate computations that don't have to be?

If the observed runtime is close expected, you will have to propose ideas to make the runtime faster.

- If we are memory bound, we should try lowering the memory loads by downcasting from `fp32` to `bf16`, or from `bf16` to `i8/i4` through quantization.
- If we are compute-bound, we have successfully saturated the chip's compute units and achieved peak throughput. At this point, you can't get more throughput. The conversation now shifts to latency. To lower latency, you must reduce the total work, which means either:
  - Reducing the batch size, which will trade some of our hard-won throughput for lower latency.
  - Reducing the FLOPS of the model itself through techniques like pruning or distillation.

## Performance Modelling

You will be shown a model's architecture, and given a chip's specification.

1. You will be asked to model the runtime of the model given different batch sizes.
   - Compute the expected runtime as we've seen before.
2. You will have to compute the amount of memory required for each batch size.
3. You will be tasked with finding the optimum batch size to maximize throughput.
   - According to the roofline model, this corresponds to the smallest batch size that makes the operation compute-bound. This is the "knee" or "ridge point" of the roofline. Any batch size larger than this will only increase latency without any corresponding gain in throughput, as you're already at `Peak_FLOPS`.
   - Find which batch size is compute bound by simply figuring out which batch size has a higher compute time than memory time.
   - The optimum batch size should not use more memory than available on the chip.
