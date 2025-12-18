# Practice: Implementing An LLM's Forward Pass

We have now covered `NumPy`'s most important APIs. Let's use them to implement the hottest model architecture: the Transformer.

Specifically, we are implementing a decoder-only transformer similar to LLAMA. We use [RoPE](https://arxiv.org/abs/2104.09864) for our positional encoding.

A major difference with LLAMA is that we use `post-norm` instead of `pre-norm`. We normalize after the attention mechanism and residual instead of before each block. We do this for convenience but you will almost never see this in real life.

## Embedding Lookup

The input to our model is a one dimensional array of integers. These integers correspond to token ids, we use these token ids to retrieve the corresponding embeddings for each token in our input.

Let's define our sequence length (the number of ids) as `256`. Our vocabulary size is `32768`, this is the total count of possible token ids, anything above this value will be incorrect. Our model dimension is `512`.

**You can edit the next snippets or copy-paste the code into a Jupyter Notebook like [Google Colab](https://colab.google.com/).**

We provide a random initialization of the input token ids and the vocab. Implement the lookup method to map the token ids to their corresponding embeddings. You click on the `solution` button below to reveal the solution.

```python,editable
import numpy as np

sequence_length = 256
vocab_size = 32768
model_dim = 512

# -- Initiate Random Values --
# (sequence_length,)
input_token_ids = np.random.randint(0, vocab_size, size=(sequence_length,))
# (vocab_size, model_dim)
vocab = np.random.normal(size=(vocab_size, model_dim)).astype(np.float16)

def embeddings_lookup(input_ids, vocab) -> np.ndarray:
    # -- Your Code --
    ...
```

<details>
    <summary>Solution</summary>

```python

def embeddings_lookup(input_ids, vocab) -> np.ndarray:
    return vocab[input_ids]

# (sequence_length, model_dim)
embeddings = embeddings_lookup(input_token_ids, vocab)
```

</details>

## Attention Mechanism

Now that we have our embeddings, we pass them through the multi-head attention mechanism. It is the crux of the transformer architecture.

### Q,K,V projections

The first thing we need to do is multiply our token embeddings with the trained \\(Q\\), \\(K\\), \\(V\\) weights. Since we are using the multi-head attention architecture, the weights are split into `num_heads` heads of shape `head_dim`. They all share the same shape `(model_dim, num_heads, head_dim)` so that they can be multiplied to our input embeddings.

In the next snippet, we initialize the weights, and you implement the \\(Q\\), \\(K\\), \\(V\\) projections.

```python,editable
num_heads = 4
head_dim = 64

attn_shape = (model_dim, num_heads, head_dim)

# -- Initiate Random Values --

q_weights = np.random.normal(size=attn_shape).astype(np.float16)
k_weights = np.random.normal(size=attn_shape).astype(np.float16)
v_weights = np.random.normal(size=attn_shape).astype(np.float16)

def qkv_proj(
    embeddings, q_weights, k_weights, v_weights
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # -- Your Code --
    ...

```

<details>
    <summary>Solution</summary>

```python

def qkv_proj(
    embeddings, q_weights, k_weights, v_weights
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  q = np.einsum('sd,dnh->snh', embeddings, q_weights)
  k = np.einsum('sd,dnh->snh', embeddings, k_weights)
  v = np.einsum('sd,dnh->snh', embeddings, v_weights)
  return q, k, v

qkv_proj(embeddings, q_weights, k_weights, v_weights)
```

</details>

### RoPE

At this point, our model has no way of knowing in which order the tokens appeared in the sequence. Furthermore, the attention mechanism does not inherently expose this information. Therefore, we need to tweak our embeddings according to their position in the sequence.

[RoPE](https://arxiv.org/abs/2104.09864) is a very common approach. It introduces an efficient trick to apply a rotation matrix with varying angles depending on the index in the sequence and the index in the vector.

We use the `split` variant for convenience. A common alternative is the `interleaved` variant.

\\(RoPE\\) encodes position information by rotating pairs of query and key vectors in a 2D plane. For a vector \\(x\\) at position \\(m\\), the rotated vector is computed as: $$\text{RoPE}(x, m) = x \cdot \cos(m\theta) + \text{rotate\_half}(x) \cdot \sin(m\theta)$$

Where \\(\text{rotate\_half}\\) swaps the components of pairs and negates the first one: \\[\text{rotate\_half} \begin{pmatrix} x_1 \\\ x_2 \end{pmatrix} = \begin{pmatrix} -x_2 \\\ x_1 \end{pmatrix}\\]

\\[\text{RoPE}(x, m) = x \cdot \cos(m\theta) + \text{rotate\_half}(x) \cdot \sin(m\theta)\\]

*Let's implement \\(RoPE\\), we provide the code to generate \\(\theta\\).*

```python,editable
def apply_rotary_emb(x):
    dim = x.shape[-1]
    # 1. Generate Theta
    theta = 1.0 / (10_000 ** (np.arange(0, dim, 2) / dim))

    # 2. Generate the positions m (indices of the tokens)

    # 3. Multiply all indices with all theta (outer product of m and theta)

    # 4. Apply cos and sin (separately) to outer product

    # 5. Split x's last axis in 2 (we call the first half x1, the second x2)

    # 6. We will now combine the output
    # The first half (out1) is x1 * cos - x2 * sin
    # The second half is x1 * sin + x2 * cos
    # Think about broadcasting cos and sin first

    # 7. Return the concatenation of out1 and out2 (np.concatenate)

# 8. Apply to q and k
q = apply_rotary_emb(q)
k = apply_rotary_emb(k)
```

<details>
    <summary>Solution</summary>

```python
def apply_rotary_emb(x):
    dim = x.shape[-1]
    # 1. Generate Theta
    theta = 1.0 / (10_000 ** (np.arange(0, dim, 2) / dim))

    # 2. Generate the positions m (indices of the tokens)
    m = np.arange(x.shape[0])

    # 3. Multiply all indices with all theta (outer product of m and theta)
    freqs = m[:, None] * theta[None, :]

    # 4. Apply cos and sin (separately) the the outer product
    cos = np.cos(freqs)
    sin = np.sin(freqs)

    # 5. Split x's last axis in 2 (we call the first half x1, the second x2)
    x1 = x[..., :dim // 2]
    x2 = x[..., dim // 2:]

    # 6. We will now combine the output
    # The first half (out1) is x1 * cos - x2 * sin
    # The second half is x1 * sin + x2 * cos
    # Think about broadcasting cos and sin first
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # 7. Return the concatenation of out1 and out2 (np.concatenate)
    return np.concatenate([out1, out2], axis=-1)

# 8. Apply to q and k
q = apply_rotary_emb(q)
k = apply_rotary_emb(k)
```

</details>

### Attention Scores

We have encoded all the necessary information into our \\(q\\) and \\(k\\) tensors. We can now multiply them together and apply `softmax` to the output.

\\[\text{AttentionScores}(Q, K) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)\\]

This is the most crucial part of the attention mechanism. To understand, let's take a look at the dimensions. \\(q\\) and \\(k\\) are both of shape `(sequence, num_heads, head_dim)`. We contract the head dimension (`head_dim`) which means that our output will be of shape `(sequence, num_heads, sequence)`. So we get `sequence` twice in our output. This means that we get a score for each \\(q\\) to each \\(k\\). When we normalize with `softmax`, the sum of the scores adds up to `1`.

The last missing piece of the `Attention Scores` is the `masking`. A token cannot have access to a token that appeared after it in the sequence otherwise it would have access to the future. To remedy this, we simply mask the scores before the softmax. The mask looks like this:

\\[M = \begin{bmatrix}
0 & -\infty & -\infty\\\ 0 & 0 & -\infty \\\ 0 & 0 & 0
\end{bmatrix}\\]

The equation becomes

\\[\text{AttentionScores}(Q, K) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right)\\]

Why \\(-\infty\\)? When we apply the exponential function during softmax (\\(e^{-\infty}\\)), the result becomes `0`.

```python,editable
def softmax(x):
    # Subtract max to prevent overflow
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def attention_scores(q, k):
    # 1. Multiply q and k
    # Output shape: (Seq_q, Batch, Seq_k)

    # 2. Divide by the square root of head_dim

    # 3. Generate the mask
    
    # 4. Apply the mask

    # 5. Return softmax
```

<details>
    <summary>Solution</summary>

```python
def softmax(x):
    # Subtract max to prevent overflow
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def attention_scores(q, k):
    # 1. Multiply q and k
    qk = np.einsum('snh,tnh->snt', q, k)

    # 2. Divide by the square root of head_dim
    qk /= np.sqrt(q.shape[-1])

    # 3. Generate the mask
    seq = q.shape[0]
    bool_mask = np.arange(seq)[:, None] >= np.arange(seq)[None, :]
    mask = np.where(bool_mask, 0, -np.inf)
    
    # 4. Apply the mask
    qk += mask[:, None, :]

    # 5. Return softmax
    return softmax(qk)
```
</details>

### qk @ v

We have scores for each key, query pair. We multiply the scores with the values. Since the scores add up to `1`, we are essentially doing a weighted average of the values depending on the score for each key.

```python,editable
# Multiply qk and v
```

<details>
    <summary>Solution</summary>

```python
qkv = np.einsum('snt,tnh->snh', qk, v)
```
</details>

### Final Attention Projection

Our activations now have shape `(sequence, num_heads, head_dim)`. We want to go back to `model_dim` before applying the MLP. So we project back with learnt weights.

```python,editable
upproj_weights = np.random.normal(size=(num_heads, head_dim, model_dim)).astype(np.float16)

# Project qkv with upproj_weights
```

<details>
    <summary>Solution</summary>

```python
upproj_weights = np.random.normal(size=(num_heads, head_dim, model_dim)).astype(np.float16)

attention_out = np.einsum('snh,nhd->sd', qkv, upproj_weights)
```

</details>

### Residual and Normalization

Finally, for better gradient flow and to constrain the latent space, we apply a residual connection. We simply add the output of the attention mechanism to the original input. Furthermore, we normalize the output to prevent the gradients from exploding.

In real LLMs, we usually normalize before attention, and then again before MLP. We did it this way for convenience.

\\[x_{out} = \text{RMS\_Norm}(x + \text{Attention}(x))\\]

**The Formula:**
\\[x_{norm} = \frac{x}{\text{RMS}(x)} \cdot \gamma\\]

Where:
\\[\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}\\]

```python,editable
# 1. Residual (add the output of attention back to the original input)

def rms_norm(x, gamma=1.0):
    # 2. Implement rms square root of the sum of the squares of x
    # on the last dimension

# 3. Apply RMS
```

<details>
    <summary>Solution</summary>

```python
# 1. Residual (add the output of attention back to the original input)
x = embeddings
x += attention_out

def rms_norm(x, gamma=1.0):
    # 2. Implement rms square root of the sum of the squares of x
    # on the last dimension
    d_model = x.shape[-1]
    rms = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True) / d_model)
    return gamma * x / rms

# 3. Apply RMS
x = rms_norm(x)
```

</details>

## Multi Layer Perceptron (MLP)

After attention, we run a (usually) 2 layers MLP separated by an activation function in between. Here, we run a classic \\[x + \text{ReLU}(x W_0) W_1\\]

We also run another normalization after. Again, this would typically be before in a real world use case.

**Where:**

\\[ReLu(x) = \max(0, x)\\]

We introduce a new dimension, the hidden dimension (`hidden_dim`) that we set to `4 * model_dim`

```python,editable
hidden_dim = 4 * model_dim

w0 = np.random.normal(size=(model_dim, hidden_dim)).astype(np.float16)
w1 = np.random.normal(size=(hidden_dim, model_dim)).astype(np.float16)

def mlp(x, w0, w1):
    # Your code.

x = mlp(x, w0, w1)
x = rms_norm(x)
```

<details>
    <summary>Solution</summary>

```python
hidden_dim = 4 * model_dim

w0 = np.random.normal(size=(model_dim, hidden_dim)).astype(np.float16)
w1 = np.random.normal(size=(hidden_dim, model_dim)).astype(np.float16)

def mlp(x, w0, w1):
    y = np.einsum('bd,df->bf', x, w0)
    y = np.maximum(y, 0)
    return np.einsum('bf,fd->bd', y, w1) + x

x = mlp(x, w0, w1)
x = rms_norm(x)
```

</details>

## Putting it all together

<details>
<summary>
Click to expand
</summary>

```python
import numpy as np

sequence_length = 256
vocab_size = 32768
model_dim = 512

# - EMBEDDINGS -

# (sequence_length,)
input_token_ids = np.random.randint(0, vocab_size, size=(sequence_length,))
# (vocab_size, model_dim)
vocab = np.random.normal(size=(vocab_size, model_dim)).astype(np.float16)

def embeddings_lookup(input_ids, vocab) -> np.ndarray:
    return vocab[input_ids]

# (sequence_length, model_dim)
embeddings = embeddings_lookup(input_token_ids, vocab)




# - ATTENTION -
# -- Attention Projections --
num_heads = 4
head_dim = 64

attn_shape = (model_dim, num_heads, head_dim)


q_weights = np.random.normal(size=attn_shape).astype(np.float16)
k_weights = np.random.normal(size=attn_shape).astype(np.float16)
v_weights = np.random.normal(size=attn_shape).astype(np.float16)


def qkv_proj(
    embeddings, q_weights, k_weights, v_weights
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  q = np.einsum('sd,dnh->snh', embeddings, q_weights)
  k = np.einsum('sd,dnh->snh', embeddings, k_weights)
  v = np.einsum('sd,dnh->snh', embeddings, v_weights)
  return q, k, v

q, k, v = qkv_proj(embeddings, q_weights, k_weights, v_weights)

# -- RoPE --

def apply_rotary_emb(x):
    dim = x.shape[-1]
    # 1. Generate Theta
    theta = 1.0 / (10_000 ** (np.arange(0, dim, 2) / dim))

    # 2. Generate the positions m (indices of the tokens)
    m = np.arange(x.shape[0])

    # 3. Multiply all indices with all theta (outer product of m and theta)
    freqs = m[:, None] * theta[None, :]

    # 4. Apply cos and sin (separately) the the outer product
    cos = np.cos(freqs)
    sin = np.sin(freqs)

    # 5. Split x's last axis in 2 (we call the first half x1, the second x2)
    x1 = x[..., :dim // 2]
    x2 = x[..., dim // 2:]

    # 6. We will now combine the output
    # The first half (out1) is x1 * cos - x2 * sin
    # The second half is x1 * sin + x2 * cos
    # Think about broadcasting cos and sin first
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    # 7. Return the concatenation of out1 and out2 (np.concatenate)
    return np.concatenate([out1, out2], axis=-1)

# 8. Apply to q and k
q = apply_rotary_emb(q)
k = apply_rotary_emb(k)


# -- Attention Scores --

def softmax(x):
    # Subtract max to prevent overflow
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def attention_scores(q, k):
    # 1. Multiply q and k
    qk = np.einsum('snh,tnh->snt', q, k)

    # 2. Divide by the square root of head_dim
    qk /= np.sqrt(q.shape[-1])

    # 3. Generate the mask
    seq = q.shape[0]
    bool_mask = np.arange(seq)[:, None] >= np.arange(seq)[None, :]
    mask = np.where(bool_mask, 0, -np.inf)
    
    # 4. Apply the mask
    qk += mask[:, None, :]

    # 5. Return softmax
    return softmax(qk)

qk = attention_scores(q, k)

# -- qk @ v --

qkv = np.einsum('snt,tnh->snh', qk, v)

# -- Final Attention Proj --

upproj_weights = np.random.normal(size=(num_heads, head_dim, model_dim)).astype(np.float16)
attention_out = np.einsum('snh,nhd->sd', qkv, upproj_weights)

# -- Residual and RMS norm --

x = embeddings
x += attention_out

def rms_norm(x, gamma=1.0):
    # 2. Implement rms square root of the sum of the squares of x
    # on the last dimension
    d_model = x.shape[-1]
    rms = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True) / d_model)
    return gamma * x / rms

# 3. Apply RMS
x = rms_norm(x)


# - MLP -
hidden_dim = 4 * model_dim

w0 = np.random.normal(size=(model_dim, hidden_dim)).astype(np.float16)
w1 = np.random.normal(size=(hidden_dim, model_dim)).astype(np.float16)

def mlp(x, w0, w1):
    y = np.einsum('bd,df->bf', x, w0)
    y = np.maximum(y, 0)
    return np.einsum('bf,fd->bd', y, w1) + x

x = mlp(x, w0, w1)
x = rms_norm(x)
```

</details>
