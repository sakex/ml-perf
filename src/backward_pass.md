# Backward Pass

ML frameworks implement auto differentiation for us, so we usually do not need to implement the backward pass ourselves. Nonetheless, it is important to understand how the gradients are propagated to be able to reason about memory usage and the computational overhead of the backprop.

## What are the Forward and the Backward Pass?

The `Forward pass` is the "main formula" of the model. It is the primary computation of the model, transforming inputs to outputs. It is executed during both inference (to get predictions) and training (to compute the loss).

The `Backward Pass` (Backpropagation) is the chain rule of calculus applied in reverse. It computes the gradient of the loss function with respect to every parameter in the model. These gradients indicate the direction and magnitude to adjust each parameter to minimize error.

The `forward pass` produces `predictions`. During training, these predictions are compared to a ground truth value to calculate a `loss`. The `backward pass` produces `gradients`, or directions for each parameter in the model. The `gradients` are consumed by the `optimizer` to update the weights used in both the forward and the backward pass.

## How is it computed?

Most functions in ML are differentiable (or have defined subgradients for points like `x=0` in ReLU). Therefore, when ML frameworks developers implement a function, they implement its `forward` and `backward` methods.

A basic example using `jax.grad`, the derivative of the Sine function `sin` is simply the Cosine `cos`. When we apply `jax.grad` to `jnp.sin`, `jax` will internally call `jnp.sin.backward` (not the exact internal name), which is `jnp.cos`.

```python
import jax.numpy as jnp

grad_sin = jax.grad(jnp.sin)
grad_sin(0.2) == jnp.cos(0.2)
```

Let's take a look at a more complex example:

```python
import jax
import jax.numpy as jnp

key0, key1, key2 = jax.random.split(jax.random.key(0), 3)

b, d, f = 16, 64, 32
x = jax.random.normal(key0, (b, d))
w_0 = jax.random.normal(key1, (d, f))
w_1 = jax.random.normal(key2, (f, d))

def mlp(args):
  x, w_0, w_1 = args
  z = x @ w_0
  z_relu = jax.nn.relu(z)
  out = z_relu @ w_1
  return 0.5 * jnp.sum(out ** 2)

grad_mlp = jax.grad(mlp)

# jax.grad only returns the gradients of the first argument
# so we pass all our arguments as a tuple
grad_mlp((x, w_0, w_1))
```

This is a classic 2 layers `MLP` with a `ReLU` activation in between. `relu(x @ w0) @ w1`.

- The `Forward Pass` simply executes the code we wrote.
- The `Backward Pass` takes the output of the `Forward Pass` and executes the backward methods in reverse order by propagating gradients backward.
        - Some derivatives require the original activation from the `Forward pass` so we need to individually store them during the `forward call`.

### Walking through the backward pass

1. `0.5 * jnp.sum(out ** 2)` The derivative is simply `out`
2. `out = z_relu @ w_1` Here we need to compute the gradients of `w_1` which will be used to update `w_1` by the optimizer and the gradients of `z_relu` that will be backpropagated.
    - `dL/dW1 = z_relu.T @ grads` (`(b, f).T @ (b, d) -> (f, d)`)
    - `dL/dZ_relu = grads @ w1.T` (`(b, d) @ (f, d).T -> (b, d)`)
3. `z_relu = jax.nn.relu(z)` `ReLU` is defined as `relu(x) = max(0, x)`. Its derivative is therefore `d_relu(x) = 0 if x <= 0 else 1`. We then multiply the derivative with the gradients.
    - **Performance Note:** Storing values in `HBM` (High Bandwidth Memory) is expensive. For element-wise operations like ReLU, it is often faster to `recompute` the activation during the backward pass using the cached input (`z`) rather than storing the output (`z_relu`) and reading it back. This is known as `activation recomputation` or `rematerialization`.
4. `z = x @ w_0`. Just like for layer 1:
    - `dL/dW0 = x.T @ grads` (`(b, d).T @ (b, f) -> (d, f)`)
    - `dL/dx = grads @ w0.T` (`(b, f) @ (d, f).T -> (b, d)`)

First let's rewrite the MLP implementation to cache the intermediate activations:

```python
def mlp_activations(x, w_0, w_1):
  activations = [x]
  z = x @ w_0
  activations.append(z)
  z_relu = jax.nn.relu(z)
  activations.append(z_relu)
  out = z_relu @ w_1
  activations.append(out)
  return 0.5 * jnp.sum(out ** 2), activations
```

Now let's implement the backward pass:

```python
def manual_mlp_grad(x, w_0, w_1):
  # Forward
  _, activations = mlp_activations(x, w_0, w_1)

  # Pop out, shape (b, d)
  out = activations.pop()

  # 1. Derivative dL/dOut = out
  grads = out

  # 2. Derivative of Layer 1
  z_relu = activations.pop()
  # dL/dW1
  grads_w_1 = z_relu.T @ grads
  # dL/dZ_relu
  grads_z_relu = grads @ w_1.T

  # 3. Derivative of ReLU
  z = activations.pop()
  grads_z = jnp.where(z > 0, 1, 0) * grads_z_relu

  # 4. Derivative of Layer 0
  x = activations.pop()
  # dL/dW0
  grads_w_0 = x.T @ grads_z
  # dL/dx
  grads_x = grads_z @ w_0.T

  return grads_x, grads_w_0, grads_w_1
```

Correctness check:

```python
import numpy as np

manual_out = manual_mlp_grad(x, w_0, w_1)
jax_out = grad_mlp((x, w_0, w_1))

for manual, autograd in zip(manual_out, jax_out):
  np.testing.assert_allclose(manual, autograd)
```

## Performance Implications

### Flops

As we have seen, each matrix multiplication in the forward pass requires two matrix multiplications in the backward pass. Hence, the number of flops in the backward pass can easily be approximated as twice the flops of the forward pass.

### Memory Usage

Since we need to store the intermediate activations during the forward pass, our model requires a lot more memory during training than during inference. A common rule of thumb is that Training memory is **~3x-4x** Inference memory for the same batch size, primarily due to the need to store these intermediate activations. Furthermore, constantly writing and reading previous activations saturates the memory bandwidth which slows down prefetching of other parameters.

It is crucial to study which activations are being cached, and actively find opportunities for recomputations when appropriate. Either to free up memory or to speed up the step time.
