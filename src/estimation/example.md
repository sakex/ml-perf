# Practical Example

Let's estimate the run time of a dense MLP subblock of a transformer on an A100 Nvidia GPU.

The specs for the GPU are as followed:

| Specification | A100 40GB PCIe |
| :--- | :--- |
| FP32 | 19.5 TFLOPS |
| BFLOAT16 Tensor Core | 312 TFLOPS |
| GPU Memory Bandwidth | 1,555 GB/s |


```python
@torch.compile
def mlp(x, w1, w2, wlinear):
  x1 = x @ w1
  x1 = torch.nn.ReLU()(x1)
  x2 = x @ w2
  x = x1 * x2
  out = x @ wlinear
  return out
```

Let's say that our d_model is 4096 and our hidden dimension is 8192. Let's start with a batch size of 32.

1. **Tensore Core** We have 2 dot products `bd,df->bf` and one `bf,fd->bd`, `flops = 32 * 4096 * 8192 * 2` each. We calculate the matmul time against the 312 TFLOPS BF16 Tensor Core spec, as these are specialized for matrix operations.

```python
tc_time_secs = flops * 3 / tensor_core_flops_per_sec 
tc_time_secs = 32 * 4096 * 8192 * 2 * 3 / (312 * 1e12)
tc_time_secs = 0.00002065
```

2. **CUDA Cores** We have the ReLU and the elementwise multiplication `x1 * x2` bot of these operations take `b * f` flops. We calculate the ReLU and element-wise ops against the 19.5 TFLOPS FP32 CUDA Core spec, as Tensor Cores cannot run these.

```python
cuda_time_secs = 32 * 8192 * 2 / (19.5 * 1e12)
cuda_time_secs = 2.69e-8
```

3. **Memory load times** We have to load `x`, `w1`, `w2`, and `wlinear`. We have to write the output. We assume we do not have to write and read the intermediate activations... This is a key benefit of using `torch.compile`, which performs kernel fusion. It merges the `matmul`, `ReLU`, and `element-wise multiply` operations into a single kernel, so the intermediate results (like `x1` and `x2`) never have to be written to or read from the main (HBM) memory.

```python
bf16_bitsize = 2

x_size = 32 * 4096 * bf16_bitsize
out_size = x_size
w_size = 4096 * 8192 * bf16_bitsize
total_size = x_size + 3 * w_size + out_size

memory_time_secs = total_size / (1.555 * 1e12)
memory_time_secs = 0.000129
```

4. **Estimation**

```python
estimation = max(tc_time_secs, cuda_time_secs, memory_time_secs)
estimation = 0.000129
```

Our estimation is ~129µs. Let's run it into colab on an A100.

```python
mlp(x, w_gating_1, w_gating_2, w_linear)
%timeit mlp(x, w_gating_1, w_gating_2, w_linear)
```

```
191 µs ± 8.17 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

The actual runtime is 48% slower than our estimate, which is expected because of inefficiencies like kernel launch overheads, synchronization, and any small gaps between the fused operations.
