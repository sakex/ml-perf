# KV Caching

KV Caching is one of the most fundamental optimization techniques for LLMs. It trades off memory for latency by caching the \\(K\\) and \\(V\\) activations.

Let's take a look at this small stripped down implementation of the attention mechanism. In the einsums we have:

1. \\(s=sequence\\_length_q\\)
2. \\(t=sequence\\_length_k\\)
3. \\(d=model\\_dim\\)
4. \\(n=num\\_heads\\)
5. \\(h=head\\_dim\\)

```python
q = np.einsum('sd,dnh->snh', x, q_weights)
k = np.einsum('sd,dnh->snh', x, k_weights)
v = np.einsum('sd,dnh->snh', x, v_weights)

qk = np.einsum('snh,tnh->snt', q, k)
scores = softmax(qk, axis=-1)
out = np.einsum('snt,tnh->snh', scores, v)
```

When we first pass the tokens through our model, we need to execute this full computation. The \\(Q@K^T\\) part has quadratic complexity with regard to the sequence length (\\(O(n^2)\\).)

This first pass generates a single token. To get the next token after that, we need to add back the token we just created to the sequence (\\(x\\) here), and rerun the whole model again. We do this in a loop until we reach a special `<end>` token.

However, we notice that a lot of this work is redundant after the first step. First of all, we do not need to multiply each token in \\(q\\) again, just the one we produced in the previous step, this makes the complexity per step \\(O(n)\\) where \\(n\\) is the current sequence length.

Furthermore, we can just cache the previous \\(k\\) and \\(v\\) projections so we do not have to recompute them fully at each step. We once again only project the last produced token. This would otherwise be prohibitively expensive for large sequence lengths.

We typically call the first step with a large sequence length `Prefill` and the subsequent steps that process a single token `Decode`.

Here is some pseudcode to show how decode would be implemented:

```python
# Notice that we only attend to a single token
q_current = np.einsum('1d,dnh->1nh', x[-1:], q_weights)
k_current = np.einsum('1d,dnh->1nh', x[-1:], k_weights)
v_current = np.einsum('1d,dnh->1nh', x[-1:], v_weights)

# Store the new activations
kv_store.append(k_current, v_current)

# Get the full kv cache
k, v = kv_store.get()

# q_current @ k^T
# Attention: Compare current query (1) against all keys (t)
# Note: 't' grows by 1 at every step
qk = np.einsum('1nh,tnh->1nt', q_current, k)
scores = softmax(qk, axis=-1)
out = np.einsum('1nt,tnh->1nh', scores, v)
```

This approach raises a new problem of its own. The `Prefill` step is compute-bound, we have a large sequence to which we apply a quadratic matrix multiplication. On the flip side, the `Decode` steps are memory bound, they operate on a small batch size of 1. This is exacerbated in the MLP layers. Since we are only processing a single token (Batch Size = 1), we have to retrieve the entire, massive weight matrix from memory just to perform a single matrix-vector multiplication. We spend more time moving data than computing.

Thankfully, there is a solution for this in the next chapter on [Disaggregated Serving](./disagg.md).

![image](kv_cache.png)
