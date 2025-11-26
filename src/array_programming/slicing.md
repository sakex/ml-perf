# Slicing

Slicing allows taking a view of subset of an array. Most slicing operations will not allocate extra memory, they will create a new view into the original buffer with a different starting address, a new shape, and a different stride.

## Syntax

The API to slice an array revolves around the overloaded `indexing operator` (`[]`).

## Single Axis

For an array with a single axis, it behaves exactly like a normal python list. We can

1. Get a single scalar by specifying its index `arr[2]`
2. Use negative indexing to index from right to left `arr[-1]`
3. Use a slice object to get multiple indices (from start to end with the last index excluded.) `arr[3:7]` or `slice(3, 7)`.
4. Add a step to the slice object to how many indices to skip between two elements. `arr[3:7:2]` will get elements at indices `3` and `5`. `arr[7:3:-1]` is the reversed version of `arr[3:7]`.

*Performance Note:* When you use a step (e.g., `::2`), NumPy simply doubles the stride in the metadata. The memory is untouched.

```python
import numpy as np

arr = np.arange(10)
print(f'{arr[2]=}')
print(f'{arr[-1]=}')
print(f'{arr[3:7]=}')
print(f'{arr[slice(3, 7)]=}')
print(f'{arr[3:7:2]=}')
print(f'{arr[7:3:-1]=}')
```

*stdout*

```python
arr[2]=np.int64(2)
arr[-1]=np.int64(9)
arr[3:7]=array([3, 4, 5, 6])
arr[slice(3, 7)]=array([3, 4, 5, 6])
arr[3:7:2]=array([3, 5])
arr[7:3:-1]=array([7, 6, 5, 4])
```

## Out of Bound

- Out of bound access to a scalar is illegal, `np.arange(10)[100]` raises an `IndexError`.
- But out of bound slicing is fine `np.arange(10)[100:1]` will just return an empty array (`shape = (0,)`).

## Multiple Axes

Arrays with multiple axes can be sliced using the same mechanism:

```python
import numpy as np

arr = np.arange(10).reshape(5, 2)
print(f'{arr=}')
print(f'{arr[2]=}')
print(f'{arr[1:5]=}')
print(f'{arr[1:5:2]=}')
```

*stdout*

```python
arr=array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
arr[2]=array([4, 5])
arr[1:5]=array([[2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
arr[1:5:2]=array([[2, 3],
       [6, 7]])
```

We can also slice multiple axes at once by separating them with a coma `,`:

```python
import numpy as np

arr = np.arange(12).reshape(4, 3)
print(f'{arr=}')
print(f'{arr[2, 1]=}')
print(f'{arr[2, 1:3]=}')
print(f'{arr[2:4, 1:3]=}')
```

*stdout*

```python
arr=array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
arr[2, 1]=np.int64(7)
arr[2, 1:3]=array([7, 8])
arr[2:4, 1:3]=array([[ 7,  8],
       [10, 11]])
```

We can slice a full axis by inserting `:` in its position. If we provide less indices than we have axes, `NumPy` will automatically append `:` to the missing axes as we have seen earlier. If we just want to slice the last indices and take a full view of the first ones, we can use the `...` syntax (`ellipsis`.)

```python
import numpy as np

arr = np.arange(12).reshape(2, 3, 2)
print(f'{arr=}')
print(f'{arr[:, -1, :]=}')
print(f'{arr[:, -1]=}')
print(f'{arr[..., -1]=}')
print(f'{arr[..., -1, -1]=}')
```

*stdout*

```python
arr=array([[[ 0,  1],
        [ 2,  3],
        [ 4,  5]],

       [[ 6,  7],
        [ 8,  9],
        [10, 11]]])

arr[:, -1, :]=array([[ 4,  5],
       [10, 11]])

arr[:, -1]=array([[ 4,  5],
       [10, 11]])

arr[..., -1]=array([[ 1,  3,  5],
       [ 7,  9, 11]])

arr[..., -1, -1]=array([ 5, 11])
```

## Mutation

Since a slice is just a window into the same memory, **modifying the slice modifies the original array.**

```python
import numpy as np

original = np.zeros(5)
slice_view = original[0:2]

# Modify the slice
slice_view[:] = 100

print(f'{original=}') # Original is changed!
```

*stdout*

```python
original=array([100., 100.,   0.,   0.,   0.])
```

If you need to modify a slice without affecting the original, you must explicitly call `.copy()`.
