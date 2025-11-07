# How to Estimate our Performance?

We need to figure out which part of the chip will be taking the largest amount of time.

Therefore, we need:

1. How much time will it take the ALUs to execute all computations?
2. How long will we spend loading the data from the main memory to the ALUs?
3. What is the maximum of these two values?

Let's start with 1. Typically, we will estimate a simple dot product. The flops of a dot product are computed as such:

```python
for an mk,kn->mn dot product

flops = m * k * n * 2
```

Now we need to divide the number of flops by the theoretical limit of the machine to get the peak theoretical compute performance.

```python
compute_seconds = flops / flops_per_second(chip)
```

Now, let's compute the time it will take to load the memory onto the ALUs and to write the output back.

```python
let's call the left hand side "lhs" and the right hand side "rhs".

total_memory_bytes = (m * k * bytes_per_element_lhs) + (k * n * bytes_per_element_rhs) + (m * n * bytes_out)
```

The time it should take to load the memory will be

```python
memory_time_seconds = total_memory_bytes / memory_bandwidth_seconds(chip) 
```

Our theoretical run time will be:

```python
max(compute_seconds, memory_time_seconds)
```
