# Buffer

## Overview
In deep learning, we often work with large amounts of data organized in multi-dimensional arrays, such as images or sequences. To perform efficient computations on this data, we need a flexible and powerful data structure that can handle:
  - __Storage__: Efficiently store multi-dimensional data in contiguous or non-contiguous memory layouts.
  - __Manipulation__: Support a wide range of mathematical operations like addition, multiplication, exponentiation, and more.
  - __Memory Management__: Handle complex indexing, slicing, and broadcasting operations while minimizing memory overhead.
  - __Abstraction__: Provide a layer of abstraction over raw data, enabling operations like reshaping, expanding, and permuting dimensions.

The `Buffer` class addresses these needs by offering a versatile way to manage and manipulate multi-dimensional arrays of data in TinyGPT. It serves as the backbone for tensors and other high-level abstractions in the library.

You can find the implementation of the Buffer class in the TinyGPT source code [here](../src/tinygpt/buffer.py).

## Data storage
The `Buffer` class stores data in a one-dimensional (1D) array, regardless of how many dimensions the original data has. This flattened storage approach allows for more efficient memory usage and manipulation, as it simplifies the underlying data management. However, to interact with this 1D data as if it were multi-dimensional, Buffer uses additional metadata: __shape__, __stride__, __offset__, and __dtype__.

### Shape

The shape of a `Buffer` defines the size of each dimension of the multi-dimensional array. For example, a 3x3 matrix has a shape of `(3, 3)`, indicating that it has 3 rows and 3 columns.

### Stride

Stride indicates how many elements in the 1D data array you need to skip to move from one element to the next along a particular dimension. Stride is crucial for determining how multi-dimensional arrays are mapped to and from the 1D array.

### Offset

The offset specifies where in the 1D data array the buffer’s actual data begins. This is particularly useful for operations like slicing, where different buffers might share the same underlying data array but start at different points.

### Data Type (DType)

Every buffer in TinyGPT is associated with a specific data type, known as `DType`. This data type determines the kind of elements that the buffer can store, such as integers, floats, or booleans. The `DType` is crucial for ensuring that operations on buffers are performed correctly.

The following data types are supported:
  - `DType.float32`: 32-bit floating-point numbers, commonly used for neural network weights and activations.
  - `DType.int32`: 32-bit integers, useful for indexing or discrete data.
  - `DType.bool`: Boolean values, typically used for masks or logical operations.

You can find the implementation of the `DType` class in the TinyGPT source code [here](../src/tinygpt/utils.py).

### Example: Storing a 2D Array in a Buffer
Let's consider a 2D array:

```python
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

This array will have the following properties when stored in a `Buffer` object:

  - __1D Data Storage__: This array would be stored as a flat list: `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.

  - __Shape__: The shape of the buffer would be `(3, 3)`.

  - __Stride__: To traverse the rows, you move by a stride of 3 (since there are 3 columns). The stride for the columns is 1 (since each element within a row is next to the other). So, the stride for this buffer would be `(3, 1)`.

  - __Offset__: If the buffer starts at the beginning of the array, the offset is `0`. If it starts at a different position, the offset would adjust accordingly.

  - __DType__: If this buffer is storing 32-bit integer numbers, its dtype would be `DType.int32`.

Using these concepts, `Buffer` allows efficient and flexible manipulation of data, enabling operations like reshaping (changing the shape without altering the data), slicing (creating sub-arrays), and broadcasting (performing operations on arrays of different shapes).

If you want additional help to grasp these concepts, I recommend visiting this excellent resource: [Stride Guide by ajcr](https://ajcr.net/stride-guide-part-1/).

## What Can You Do with a Buffer
The `Buffer` class offers a rich set of functionalities that allow you to perform various operations on multi-dimensional data. Below, we mention only some of these functionalities as examples; however, `Buffer` implement more. We encourage you to explore the source code to discover all the capabilities that `Buffer` provides.

### Creation and Initialization
  - __Create Buffers__:
    - Buffers can be initialized with specific shapes and data types using convenient factory methods such as `Buffer.uniform()`, `Buffer.normal()`, `Buffer.zeros()`, and `Buffer.ones()`. These methods allow you to easily create buffers filled with uniformly distributed random values, normally distributed random values, zeros, or ones, respectively.

  - __Convert Data Structures__:
    - The `Buffer` constructor allows you to seamlessly convert nested lists or tuples into flat arrays while automatically deducing the appropriate data type (`DType`). 

### Mathematical Operations

  - __Element-wise Operations__:
    - Perform element-wise operations such as addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), and exponentiation (`**`). These operations can be applied between buffers of the same shape, or between a buffer and a scalar.

  - __Mathematical Functions__:
    - Apply mathematical functions like logarithm (`log()`), exponentiation (`exp()`), and maximum (`maximum()`). These functions operate on each element of the buffer individually, returning a new buffer with the results.

  - __Comparison Operations__:
    - Support for comparison operations like less than (`<`), less than or equal to (`<=`), greater than (`>`), greater than or equal to (`>=`), equality (`==`), and inequality (`!=`). These operations return a new buffer with boolean values indicating the result of the comparison for each element.

### Indexing and Slicing
  - __Access and Modify Elements__:
    - Access and modify specific elements, slices, or sub-arrays of the buffer using standard Python indexing and slicing syntax. For example, you can access a single element with `buffer[0, 1]`, or a sub-array with `buffer[0:2, :]`.

### Movement Operations
  - __Reshape__:
    - Change the shape of the buffer without altering its underlying data using the `reshape()` method. For example, you can reshape a buffer from a 1D array of shape `(9,)` to a 2D array of shape `(3, 3)`.

  - __Expand__:
    - Expand the dimensions of the buffer using the `expand()` method. This allows you to increase the size of dimensions with size 1 to match a desired shape, enabling __broadcasting__ in operations. For example, if you have a buffer of shape `(1, 3)`, you can expand it to shape `(4, 3)`.

  - __Permute__:
    - Rearrange the dimensions of the buffer using the `permute()` method, which allows you to change the order of dimensions. For example, if you have a buffer of shape `(2, 3, 4)`, you can permute it to shape `(4, 2, 3)`.

### Reduction Operations
  - __Sum__:
    - Reduce the dimensions of the buffer by summing elements along specified axes using the `sum()` method. For example, summing a buffer of shape `(2, 3)` along axis 0 will reduce it to shape `(3,)`.

  - __Maximum__:
    - Similarly, reduce the dimensions of the buffer by finding the maximum elements along specified axes using the `max()` method.

### Utility Functions and Attributes
The Buffer class includes several utility functions and attributes that provide additional functionality for managing and understanding the data within the buffer:

  - `data`: An attribute that stores the actual data of the buffer as a flat array. 

  - `numel`: An attribute that returns the total number of elements in the buffer.

  - `shape`: An attribute that returns the shape of the buffer as a tuple.

  - `stride`: An attribute that returns the strides of the buffer. 

  - `offset`: An attribute that returns the offset in the buffer’s data array where the actual data begins.

  - `is_contiguous()`: A method that checks if the buffer’s data is stored contiguously in memory.

  - `to_python()`: Converts the buffer's data back to standard Python types, such as nested lists or scalars.

### Example Usage
Here’s an example that demonstrates some of these operations:

```python
from tinygpt.buffer import Buffer
from tinygpt.utils import DType

# Creating a Buffer with zeros
buffer = Buffer.zeros((3, 3), dtype=DType.float32)

# Performing element-wise addition
buffer = buffer + 5.0

# Accessing and modifying elements
buffer[1, 1] = 10.0

# Reshaping the buffer
reshaped_buffer = buffer.reshape((9,))

# Summing along an axis
summed_buffer = buffer.sum(axes=(0,))

# Converting to a Python list
python_list = buffer.to_python()

# Accessing the shape and stride
shape = buffer.shape
stride = buffer.stride

print("Original Buffer:\n", buffer)
print("Reshaped Buffer:\n", reshaped_buffer)
print("Summed Buffer:\n", summed_buffer)
print("Python List:\n", python_list)
print("Shape:", shape)
print("Strides:", stride)
```