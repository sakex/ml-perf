# ML Compilers

`Python` is a particularly inefficient programming language. Yet, it is used almost ubiquitously to develop massive models and deploy them efficiently at scale. *How come?*

ML frameworks circumvent `Python`'s slow runtime by compiling the model's code into machine code for the target architecture just like the `Rust` compiler would. This allows us to write efficient code **despite python**.

## High Level APIs

In this chapter, we will cover both [Jax](https://github.com/jax-ml/jax) and [PyTorch](https://pytorch.org/). They both provide an API to compile a `Python` function and make it more efficient. For now, we will only showcase the APIs. It the later subchapters, we will dive into the differences between `Jax` and `PyTorch` compilation processes and the different optimizations that the ML compilers perform.

Let's implement the attention mechanism in both `Jax` and `PyTorch` to demonstrate how the API works at a high level.

![image](ml_compilers.png)

### Jax

`Jax` offers the `jax.jit` method that takes a python function and a set of abstract inputs and compiles an optimized method for the `function, inputs` pair. Abstract inputs are composed of a `dtype` and a `shape`. The first call to the jitted function is slow because it needs to perform the compilation, the subsequent calls are very fast because the compilation is cached. Calling the same method with inputs of different `dtype` or `shape` will trigger a recompilation.

We are running this on a `TPU v6` in [Google Colab](colab.google.com). We use `block_until_ready` otherwise, `Jax` would not wait for the computations to be complete on TPU before yielding back control to the CPU.

```python
from jax import numpy as jnp
import jax

def attention(q, k, v):
  qk = jnp.einsum('btnh,bsnh->btns', q, k)
  scores = jax.nn.softmax(qk, axis=-1)
  return jnp.einsum('btns,bsnh->btnh', scores, v)

jitted_attention = jax.jit(attention)
```

#### Weights initialization

```python
key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)

shape = (32, 1024, 16, 256)

# Automatically on TPU in Jax
q = jax.random.normal(key_q, shape, dtype=jnp.bfloat16)
k = jax.random.normal(key_k, shape, dtype=jnp.bfloat16)
v = jax.random.normal(key_v, shape, dtype=jnp.bfloat16)
```

#### Runtime in Eager Mode

```python
%%time
out = attention(q, k, v).block_until_ready()
```

*stdout*

```plaintext
Wall time: 12.8 ms
```

#### First jitted call

```python
%%time
out = jitted_attention(q, k, v).block_until_ready()
```

*stdout*

```plaintext
Wall time: 1.88 s
```

#### Second jitted call

*stdout*

```plaintext
Wall time: 4.44 ms
```

### PyTorch

`PyTorch` uses the `torch.compile` method. At a high level, it is very similar to `jax.jit`. Notice how similar the code is.

We are running this on an `A100` in [Google Colab](colab.google.com).

```python
import torch

def attention(q, k, v):
  qk = torch.einsum('btnh,bsnh->btns', q, k)
  scores = torch.nn.functional.softmax(qk, dim=-1)
  return torch.einsum('btns,bsnh->btnh', scores, v)

compiled_attention = torch.compile(attention)
```

#### Weights Initialization

```python
# Explicitly set default device to GPU
device = torch.device("cuda")

shape = (32, 1024, 16, 256)

generator = torch.Generator(device=device).manual_seed(0)

q = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
k = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
v = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
```

#### Runtime Eager Mode

```python
%%time
out = attention(q, k, v)
# Equivalent of block_until_ready
torch.cuda.synchronize()
```

*stdout*

```plaintext
Wall time: 17.4 ms
```

#### First Compiled Call

```python
%%time
out = compiled_attention(q, k, v)
# Equivalent of block_until_ready
torch.cuda.synchronize()
```

*stdout*

```plaintext
Wall time: 1.61 s
```

#### Second Compiled Call

*stdout*

```plaintext
Wall time: 6.56 ms
```

## Why not just use another language?

There are efforts to create new languages for ML. For instance [Julia](https://julialang.org/) which seems to have lost its momentum and [Chris Lattner](https://en.wikipedia.org/wiki/Chris_Lattner)'s [Mojo](https://www.modular.com/mojo) which is too recent to tell.

The reasons `Python` is so commonly used in the ML community are mostly historical and cultural. The language has been around for more than 30 years, so it has a lot of mature and stable libraries that are commonly taught in universities. It is also easy to pick up and play with, making it ideal for quick iterations in research environments. At this point, `Python`'s adoption is not about its inherent qualities but mostly about network effects, which are extremely difficult to compete against.
