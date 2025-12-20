# Flash Attention

The main bottleneck of the attention mechanism comes from the `softmax` in the `attention scores` computation.

```python
scores = softmax(np.einsum('btnh,bsnh->btns', q, k))
out = np.einsum('btns,bsnh->btnh', scores, v)
```

The size of the `scores` tensor scales quadratically with regard to the sequence length \\(O(sequence\\_length ^ 2)\\). Because we apply `softmax` to the output, we materialize the whole intermediate array. We very quickly run out of memory to store this intermediate array when reaching large batch sizes.

Furthermore, the naive implementation reads the \\(QK^T\\) product twice from HBM:

1. Compute the sum of the exponentials \\(\sum_{j=1}^{n} \exp{x_j}\\)
2. Apply \\(softmax(x_i) = \frac{\exp{x_i}}{\sum_{j=1}^{n} \exp{x_j}} \\)

`Flash Attention` solves the memory issue and reduces the amount of memory to read by introducing the `Online Softmax` trick.

Instead of materializing the whole array, summing it, and dividing each value; we split \\(Q\\), \\(K\\), and \\(V\\) into sub-blocks and we compute the attention block by block. Since the `softmax` needs to know about the full sequence, we need to reconcile the values across blocks, we do this by keeping a state. Specifically:

- \\(m = max(S_i)\\). The maximum value. Used to prevent overflowing.
- \\(\ell = \sum{\exp(S_i - m_i)}\\). Normalization denominator for `softmax`.

For each block, `softmax` becomes:

\\[\text{softmax}(S_i) = \frac{e^{S_i - m}}{\ell}\\]

**The Update Rule:**
When merging a new block \\(j\\) with our running state, we update the output \\(O\\) as:

\\[O_{new} = \text{diag}(\ell_{new})^{-1} (\text{diag}(\ell_{old})e^{m_{old} - m_{new}} O_{old} + e^{m_{j} - m_{new}} P_{j} V_{j})\\]

This is scary, but it's simply the line:

```python
O_block = (O_block * old_scale) + (new_scale * (P_ij @ V_block))
```

## IO Complexity

Standard attention requires \\(O(N^2)\\) HBM accesses (to read/write the huge attention matrix). Flash Attention reduces this memory access significantly by keeping intermediate results in SRAM, though the computational complexity (FLOPs) remains quadratic.

## Code

We implement a minimal flash attention function in `numpy`. The goal is to illustrate the general logic.

Note that we would typically write a `CUDA` or `Pallas` kernel to run on a `GPU` or `TPU`. And the shape should be `(batch,seq_len,num_head,head_dim)` instead of `(seq_len,d_model)`.

```python
def flash_attention(Q, K, V, block_size=64):
    N, d = Q.shape
    scale = 1 / np.sqrt(d)
    
    # Initialize output
    O = np.zeros((N, d))
    
    # Divide Q into blocks (Rows)
    # The Outer Loop loads a block of Queries from HBM to SRAM
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        Q_block = Q[i:i_end, :]    # Shape: (Br, d)
        
        # Initialize running stats for THIS block of rows
        # shape: (Br, 1)
        m = np.full((i_end - i, 1), -np.inf)
        l = np.zeros((i_end - i, 1))
        
        # Current accumulator for this block of rows
        O_block = np.zeros((i_end - i, d))
        
        # Divide K, V into blocks (Columns)
        # The Inner Loop loads blocks of K and V from HBM to SRAM
        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            K_block = K[j:j_end, :] # Shape: (Bc, d)
            V_block = V[j:j_end, :] # Shape: (Bc, d)
            
            # --- 1. Compute Attention Scores for this sub-block ---
            # S shape: (Br, Bc)
            S_ij = (Q_block @ K_block.T) * scale
            
            # --- 2. Compute local stats for this sub-block ---
            m_ij = np.max(S_ij, axis=-1, keepdims=True) # Max of current block
            P_ij = np.exp(S_ij - m_ij)                  # Exponentials
            l_ij = np.sum(P_ij, axis=-1, keepdims=True) # Sum of curr block
            
            # --- 3. Update Global Stats (Online Softmax) ---
            # New max is max of old running max and current block max
            m_new = np.maximum(m, m_ij)
            
            # Correction factors
            # How much to shrink the old accumulator
            old_scale = np.exp(m - m_new)
            # How much to shrink the new block
            new_scale = np.exp(m_ij - m_new)
            
            # Update running sum l
            l = (l * old_scale) + (l_ij * new_scale)
            
            # Update Output Accumulator
            # O_new = O_old * scale_old + V_block * P_ij * scale_new
            O_block = (O_block * old_scale) + (new_scale * (P_ij @ V_block))
            
            # Update running max
            m = m_new

        # Finalize the block output and write to HBM
        O[i:i_end, :] = O_block / l
        
    return O
```

## Validating

```python
def standard_attention(Q, K, V):
    N, d = Q.shape
    scale = 1 / np.sqrt(d)
    
    # 1. Compute full scores matrix (N x N) - Memory intensive part!
    S = np.einsum('td,sd->ts', Q, K) * scale
    
    # 2. Compute max for numerical stability
    m = np.max(S, axis=-1, keepdims=True)
    
    # 3. Compute Softmax
    P = np.exp(S - m)
    l = np.sum(P, axis=-1, keepdims=True)
    P_norm = P / l
    
    # 4. Compute Output
    O = np.einsum('ts,sd->td', P_norm, V)
    
    return O

seq_len = 1024  # Sequence length
d = 64    # Head dimension
np.random.seed(42)

# Random Inputs
Q = np.random.randn(seq_len, d)
K = np.random.randn(seq_len, d)
V = np.random.randn(seq_len, d)

O_std = standard_attention(Q, K, V)
O_flash = flash_attention(Q, K, V, block_size=128)

# Compare
diff = np.abs(O_std - O_flash)
print(f"Max difference: {np.max(diff):.6e}")
print(f"Mean difference: {np.mean(diff):.6e}")
```
