# Indexing

We explored [slicing](./slicing.md) in the previous chapter. Building on this, we now look into indexing.

Indexing uses the same syntax as slicing, but instead of using a `slice` object or an `int`, we use another array for indexing.
While `Slicing` returns a `View` (instant, no memory cost), `Indexing` triggers a `Copy`.

## Integer Array Indexing

We can use an array of integers for indexing. `NumPy` will return the elements at the requested indices on the requested axis.

```python
import numpy as np

arr = np.arange(8).reshape(2, 4)
print(f'{arr=}')
print(f'{arr[:, np.array([0, 3])]=}')
```

*stdout*

```python
arr=array([[0, 1, 2, 3],
       [4, 5, 6, 7]])
arr[:, np.array([0, 3])]=array([[0, 3],
       [4, 7]])
```

In ML frameworks like `PyTorch` or `JAX`, this specific operation (indexing a high-dimensional tensor with a list of indices) is often called gather or take. It is expensive because the hardware must "jump around" in memory to collect the rows.

## Boolean Array Indexing (masking)

If you index using an array of `Booleans`, `NumPy` selects elements where the index is `True`.

This is widely used in ML for Filtering (e.g., for implementing `ReLU`).

*Note:* The result of boolean indexing is always a 1-D array, because the True values might not form a rectangular shape.

```python
import numpy as np

# Model predictions (logits)
logits = np.array([-1.5, 2.0, -0.1, 5.2])

# Create a boolean mask for positive values (Simulating ReLU)
mask = logits > 0

# Select only positive values
positive_activations = logits[mask]

print(f'{mask=}')
print(f'{positive_activations=}')
```

*stdout*

```python
mask=array([False,  True, False,  True])
positive_activations=array([2. , 5.2])

```

## In-Place Mutation

While extracting data (`b = a[indices]`) creates a `copy`, assigning data (`a[indices] = 0`) works in-place. This is highly efficient.

```python
import numpy as np

# Feature map
features = np.array([10, 20, 30, 40, 50])

# Indices to "drop out"
drop_indices = [0, 3]

# Modify IN PLACE (No copy created)
features[drop_indices] = 0

print(f'{features=}')
```

*stdout*

```python
features=array([ 0, 20, 30,  0, 50])
```

## Indexing vs Slicing Summary

| Operation | Syntax | Type | Memory Cost | Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Slicing** | `arr[0:5]` | **View** | ✅ **Nearly Zero** | ⚡ **Instant** |
| **Indexing** | `arr[[0, 1, 2]]` | **Copy** | ❌ **Linear `O(N)`** | 🐢 **Slower** (Memory Bound) |
