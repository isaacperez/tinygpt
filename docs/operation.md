# Operation

## Overview
In deep learning, the process of training models requires not only performing various operations on data but also calculating the gradients of these operations to update the model's parameters. While `Buffer` objects efficiently store and manipulate multi-dimensional data, they do not inherently understand how to compute the derivatives needed for __backpropagation__. This limitation is addressed by the `Operation` class in TinyGPT.

The `Operation` class allows you to define how the forward and backward passes of an operation should be computed over `Buffer` objects. Each Operation subclass in TinyGPT implements specific functionality, handling both the forward computation and the backward gradient propagation. These operations range from basic mathematical functions to more complex tasks like reshaping or permuting.

You can find the implementation of the Operation class and its subclasses in the TinyGPT source code [here](../src/tinygpt/mlops.py).

## Core Components

### Base Operation Class
The `Operation` class is an abstract base class that defines the structure for all operations in TinyGPT. It includes:
  - `needs_input_grad`: A list indicating which inputs require gradients.
  - `forward()`: A method that performs the forward computation for the operation. This method must be implemented by each subclass.
  - `backward()`: A method that computes the gradient of the operation with respect to its inputs. This method must also be implemented by each subclass.

Below are some of the key operations implemented as subclasses of `Operation`:

### Mathematical Operations

  - __Add__:
    - __Forward__: Performs element-wise addition of two buffers.
    - __Backward__: The gradient for each input is either passed through unchanged (if the input requires gradients) or set to `None`.

  - __Sub__:
    - __Forward__: Performs element-wise subtraction between two buffers.
    - __Backward__: The gradient for the first input is passed through, and the gradient for the second input is negated if it requires gradients.

  - __Neg__:
    - __Forward__: Computes the element-wise negation of a buffer.
    - __Backward__: The gradient is negated if the input requires gradients.

  - __Mul__:
    - __Forward__: Performs element-wise multiplication between two buffers.
    - __Backward__: The gradient with respect to the first buffer is computed by multiplying the incoming gradient by the second buffer, and vice versa.

  - __Div__:
    - __Forward__: Performs element-wise division of one buffer by another.
    - __Backward__: The gradients are computed considering the quotient rule for differentiation, adjusting for each input as needed.

  - __Pow__:
    - __Forward__: Raises each element of a buffer to a specified power.
    - __Backward__: The gradient is computed based on the derivative of the power function, scaled by the incoming gradient.

  - __Log__:
    - __Forward__: Computes the natural logarithm of each element in a buffer.
    - __Backward__: The gradient is computed as the incoming gradient divided by the buffer.

  - __Exp__:
    - __Forward__: Computes the exponential of each element in a buffer.
    - __Backward__: The gradient is computed by multiplying the result of the forward pass by the incoming gradient.

  - __Maximum__:
    - __Forward__: Performs element-wise maximum between two buffers.
    - __Backward__: The gradient is assigned to the buffer that held the maximum value in the forward pass.

  - __Relu__:
    - __Forward__: Applies the ReLU function, replacing all negative values in the buffer with zero.
    - __Backward__: Passes the gradient through only where the buffer was positive in the forward pass.

### Reduction Operations

  - __Sum__:
    - __Forward__: Sums the elements of a buffer along specified axes.
    - __Backward__: The gradient is expanded back to the original shape of the buffer before summing.

  - __Max__:
    - __Forward__: Computes the maximum value along specified axes of a buffer.
    - __Backward__: The gradient is distributed among the maximum values, normalized by the number of occurrences.

### Movement Operations

  - __Reshape__:
    - __Forward__: Changes the shape of a buffer without altering its underlying data.
    - __Backward__: The gradient is reshaped back to the original shape of the buffer.

  - __Expand__:
    - __Forward__: Expands the dimensions of a buffer to match a specified shape, allowing for broadcasting.
    - __Backward__: The gradient is summed along the expanded dimensions to reduce it back to the original shape.

  - __Permute__:
    - __Forward__: Rearranges the dimensions of a buffer according to a specified order.
    - __Backward__: The gradient is permuted back to the original dimension order.

### Data Manipulation Operations

  - __Slice__:
    - __Forward__: Extracts a sub-buffer based on specified indexing or slicing.
    - __Backward__: Places the incoming gradient into the correct location within the original buffer, setting all other locations to zero.

  - __Concatenate__:
    - __Forward__: Concatenates multiple buffers along a specified axis.
    - __Backward__: Splits the incoming gradient across the concatenated buffers, distributing the gradient according to the original buffer shapes.

  - __Tril__:
    - __Forward__: Extracts the lower triangular part of a buffer, setting all elements above the diagonal to zero.
    - __Backward__: Passes the gradient through only for the lower triangular part of the buffer.

## Utility and Extensibility
The `Operation` class and its subclasses provide a flexible framework for defining new operations. By extending the `Operation` class, you can create custom operations tailored to specific needs in your deep learning models.

### Implementing Your Own Operation
Here’s an example of how to implement a custom operation, such as an element-wise absolute value operation:

```python
from typing import Union

from tinygpt.buffer import Buffer
from tinygpt.mlops import Operation
from tinygpt.utils import DType

class Abs(Operation):
    def forward(self, buffer: Buffer) -> Buffer:
        # Forward pass: return the absolute value of each element in the buffer
        self.buffer = buffer
        return buffer.maximum(-buffer)

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        # Backward pass: propagate the gradient considering the sign of the input buffer
        if self.needs_input_grad[0]:
            return incoming_grad * (self.buffer > 0).float() - incoming_grad * (self.buffer < 0).float()
        else:
            return None

# Example Usage
buffer = Buffer([-1, -2, 3, 4], dtype=DType.float32)

# Instantiate the Abs operation
abs_op = Abs([True])

# Perform the forward pass
result = abs_op.forward(buffer)
print("Forward Result:\n", result)

# Perform the backward pass
incoming_grad = Buffer([1, 1, 1, 1], dtype=DType.float32)
gradients = abs_op.backward(incoming_grad)
print("Backward Gradients:\n", gradients)
```

### Example Usage
Here's how you might use some of these operations in practice:

```python
from tinygpt.buffer import Buffer
from tinygpt.mlops import Add, Mul, Sum
from tinygpt.utils import DType

# Create two Buffers
buffer1 = Buffer([[1, 2], [3, 4]], dtype=DType.float32)
buffer2 = Buffer([[5, 6], [7, 8]], dtype=DType.float32)

# Perform addition
add_op = Add([True, True])
result = add_op.forward(buffer1, buffer2)
print("Addition Result:\n", result)

# Perform multiplication
mul_op = Mul([True, True])
result = mul_op.forward(buffer1, buffer2)
print("Multiplication Result:\n", result)

# Perform summation
sum_op = Sum([True])
result = sum_op.forward(buffer1, axes=(0,))
print("Summation Result:\n", result)

# Backpropagation example
incoming_grad = Buffer([[1, 1], [1, 1]], dtype=DType.float32)
gradients = add_op.backward(incoming_grad)
print("Gradients from Addition:\n", gradients)
```