# Practice Questions

## A Practical Example

Let's explore a simple example of a post attention projection.

```python
batch = 256
length = 1024
d_model = 4096
num_heads = 16
key_size = d_model // num_heads

# B L N K
x = torch.rand((batch, length, num_heads, key_size), dtype=torch.bfloat16, device='cuda')

# N K D
w = torch.rand((num_heads, key_size, d_model), dtype=torch.bfloat16, device='cuda')

out = torch.einsum('blnk,nkd->bld', x, w)
```

1. How would pipelining work here (assuming multiple layers)?
2. How would data parallelism work?
3. What different ways can we implement tensor parallelism?
4. How would we implement FSDP?
