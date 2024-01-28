from __future__ import annotations
from typing import Any, Union

from tinygpt.utils import DType, print_dag
from tinygpt.buffer import Buffer
import tinygpt.mlops as mlops


class Tensor():

    def __init__(self, data: Any, dtype: DType = None, requires_grad: bool = False) -> None:
        # Save the data in a buffer
        self.buffer = Buffer(data, dtype)

        # Gradient-related metadata
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

        self._pending_gradients_count = 0
        self._accumulated_gradient_to_propagate = None

        if self.requires_grad and self.dtype != DType.float32:
            raise RuntimeError("Only float32 Tensors can require gradients")

    def __repr__(self) -> str:
        return (
            f"<Tensor {hex(id(self))}: {self.buffer}, shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}>"
        )

    def __str__(self) -> str:
        return f"<Tensor {self.buffer}, shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad}>"

    def print_dag(self) -> None:
        print_dag(self)

    @property
    def shape(self) -> tuple:
        return self.buffer.shape

    @property
    def ndim(self) -> int:
        return self.buffer.ndim

    @property
    def dtype(self) -> DType:
        return self.buffer.dtype

    def _broadcasted(self, other: Tensor) -> tuple[Tensor, Tensor]:
        # Make two tensors broadcastable by reshaping and expanding them as needed
        x, y = self, other

        # Return immediately if shapes are already the same
        if x.shape == y.shape:
            return x, y

        # Adjust the number of dimensions to match
        shape_delta = x.ndim - y.ndim
        if shape_delta > 0:
            # Add extra dimensions to y if x has more dimensions
            y = y.reshape((1,) * shape_delta + y.shape)
        elif shape_delta < 0:
            # Add extra dimensions to x if y has more dimensions
            x = x.reshape((1,) * -shape_delta + x.shape)

        # Check if shapes match after adding dimensions
        if x.shape == y.shape:
            return x, y

        # If shapes still don't match, expand dimensions where sizes differ
        shape_ret = tuple([max(x, y) for x, y in zip(x.shape, y.shape)])
        if x.shape != shape_ret:
            x = x.expand(shape_ret)
        if y.shape != shape_ret:
            y = y.expand(shape_ret)

        return x, y

    def __add__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Add, *self._broadcasted(other))

    def __sub__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Sub, *self._broadcasted(other))

    def __neg__(self):
        return apply_op(mlops.Neg, self)

    def __mul__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Mul, *self._broadcasted(other))

    def __truediv__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Div, *self._broadcasted(other))

    def __pow__(self, exponent: Union[int, float]) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError("Only supporting int/float powers for now")

        return apply_op(mlops.Pow, self, exponent=exponent)

    def exp(self) -> Tensor:
        return apply_op(mlops.Exp, self)

    def log(self) -> Tensor:
        return apply_op(mlops.Log, self)

    def maximum(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Maximum, *self._broadcasted(other))

    def relu(self) -> Tensor:
        return apply_op(mlops.Relu, self)

    def sum(self, axes: tuple, keepdim: bool = False) -> Tensor:
        if keepdim:
            return apply_op(mlops.Sum, self, axes=axes)
        else:
            result = apply_op(mlops.Sum, self, axes=axes)
            return result.reshape(tuple(val for val in result.shape if val != 1))

    def max(self, axes: tuple, keepdim: bool = False) -> Tensor:
        if keepdim:
            return apply_op(mlops.Max, self, axes=axes)
        else:
            result = apply_op(mlops.Max, self, axes=axes)
            return result.reshape(tuple(val for val in result.shape if val != 1))

    def softmax(self, axis: int) -> Tensor:
        if not isinstance(axis, int):
            raise TypeError(f"Expecting int value for axis but found type {type(axis)}")

        if self.ndim == 0:
            raise ValueError("Softmax not implemented for 0-d tensors")

        self_normalized = self - self.max(axes=(axis,), keepdim=True)
        self_exponential = self_normalized.exp()
        summation = self_exponential.sum(axes=(axis,), keepdim=True)

        return self_exponential / summation

    def reshape(self, shape: tuple) -> Tensor:
        return apply_op(mlops.Reshape, self, new_shape=shape)

    def expand(self, shape: tuple) -> Tensor:
        return apply_op(mlops.Expand, self, new_shape=shape)

    def permute(self, dims: tuple) -> Tensor:
        return apply_op(mlops.Permute, self, dims=dims)

    def transpose(self, axis1: int, axis2: int) -> Tensor:
        # Transpose method to swap two dimensions (axes) of a Tensor (both axes can be the same, but it has no effect)

        # Validate that axis1 and axis2 are integers
        if not isinstance(axis1, int) or not isinstance(axis2, int):
            raise TypeError("axis1 and axis2 must be an integer")

        # Check that axis1 and axis2 are within the valid range of the tensor's dimensions
        if any(axis < 0 or axis >= self.ndim for axis in [axis1, axis2]):
            raise ValueError("axis1 and axis2 should be positive and within the dimension range")

        # Generate a sequence representing the order of dimensions
        order = list(range(self.ndim))

        # Swap the positions of axis1 and axis2 in this order
        order[axis1], order[axis2] = order[axis2], order[axis1]

        # Apply the permute operation to reorder the dimensions of the tensor according to the new order
        return self.permute(tuple(order))

    def dot(self, other: Tensor):
        # Perform matrix multiplication (dot product) between this tensor (`self`) and another tensor (`other`)

        # Check if the other operand is a tensor
        if not isinstance(other, Tensor):
            raise TypeError(f"Expecting Tensor, but found type {type(other)}")

        # Ensure both tensors are at least 1-dimensional, as matrix multiplication is not defined for 0D tensors
        if self.ndim == 0 or other.ndim == 0:
            raise RuntimeError(
                f"Both arguments to matmul need to be at least 1D, but they are {self.ndim}D and {other.ndim}D"
            )

        # Check if the inner dimensions are compatible. In matrix multiplication, the number of columns in the first
        # matrix (last dimension of `self`) must be equal to the number of rows in the second matrix
        # (appropriate dimension of `other`)
        if self.shape[-1] != other.shape[-min(other.ndim, 2)]:
            raise RuntimeError(
                f"Input Tensor shapes {self.shape} and {other.shape} cannot be multiplied"
                f"({self.shape[-1]} != {other.shape[-min(other.ndim, 2)]})")

        # Reshape the tensors to make their dimensions compatible for element-wise multiplication
        # This involves adding a singleton dimension before the last dimension of `self` and before the last two
        # dimensions of `other`. It aligns the 'row' dimension of `self` with a new singleton dimension in `other`,
        # and vice versa for the 'col' dimension. This step is necessary to enable broadcasting during the element-wise
        # multiplication.
        x = self.reshape((*self.shape[0:-1], *[1] * min(self.ndim - 1, other.ndim - 1, 1), self.shape[-1]))
        w = other.reshape(
            (*other.shape[0:-2], *[1] * min(self.ndim - 1, other.ndim - 1, 1), *other.shape[-min(other.ndim, 2):])
        )

        # Transpose the last two dimensions of `w` to align it properly with `x` for element-wise multiplication
        dims_of_w = list(range(w.ndim))
        w = w.transpose(dims_of_w[-1], dims_of_w[-min(w.ndim, 2)])

        # Perform element-wise multiplication of the two tensors. This mimics matrix multiplication by multiplying
        # corresponding elements in the aligned dimensions
        mult = x * w

        # Sum over the last dimension to collapse the result into the final matrix multiplication output. This step
        # effectively sums the products of corresponding elements, completing the matrix multiplication process
        return (mult).sum((mult.ndim - 1,))

    @staticmethod
    def uniform(shape: tuple, **kwargs):
        return Tensor(Buffer.uniform(shape), **kwargs)

    def backward(self, incoming_gradient=None) -> None:
        """
        Perform the backward pass to compute gradients.

        Steps in the backward pass:
        1. Propagation of reference signal: This initial step sends a notification signal to each tensor and operation 
           involved in the computational graph. This signal helps tensors count the number of operations they are 
           involved in. This information is crucial because a tensor should not send its gradient back through the graph 
           until it has received all expected gradients from subsequent operations. Accumulating gradients is crucial 
           to avoid repeatedly traversing the graph back to the leaf nodes every time a new gradient is received.

        2. Gradient accumulation: The incoming gradient is combined with any existing gradients the tensor may have.
           This is especially important in scenarios where a tensor contributes to multiple operations, as it needs to 
           accumulate gradients from all of them before sending them backward.

        3. Gradient propagation: Once a tensor has received all expected gradients (as determined in step 1), it 
           sends its accumulated gradient backward through the graph. This step involves calling the backward method of 
           the `GradientFunction` associated with the tensor, which further propagates the gradient to the tensor's 
           inputs.
        """
        # Check if the tensor requires gradient computation
        if self.requires_grad:
            
            # Step 1: Propagate reference signal through the graph
            self._propagate_reference_signal()

            # Step 2: Initialize and accumulate the incoming gradient
            incoming_gradient = self._initialize_incoming_gradient(incoming_gradient)
            self._accumulate_gradient(incoming_gradient)

            # Step 3: Propagate the gradient backward through the graph
            self._propagate_gradient()

    def _initialize_incoming_gradient(self, incoming_gradient: Buffer) -> Buffer:
        # Initialize the incoming gradient for backward pass
        if incoming_gradient is None:
            # Handle the first backward call
            if self.ndim == 0:
                # Initialize the gradient as 1.0 for scalar tensors
                return Buffer(1.0)
            else:
                # Backward on non-scalar tensors must have an incoming gradient
                raise RuntimeError("backward can only be called on scalar tensors or with an existing gradient")
        else:
            # Handle subsequent backward calls
            # The backward method has been called from an operation in which this tensor participated and we are now
            # receiving a gradient from that operation

            # User may set incoming_gradient to something else
            if not isinstance(incoming_gradient, Buffer):
                raise TypeError("incoming_gradient is not a Buffer")

            # Use the incoming gradient provided by the operation
            return incoming_gradient

    def _accumulate_gradient(self, incoming_gradient: Buffer) -> None:
        """
        Accumulate the incoming gradient in preparation for backpropagation.

        This function updates two key attributes:
        - `grad`: This attribute accumulates the total gradient for the current tensor. It represents the tensor's 
        gradient that has been calculated so far in the backpropagation process. If the tensor is involved in 
        multiple operations, `grad` will be the sum of gradients from all these operations.
        - `_accumulated_gradient_to_propagate`: This is used to store the accumulated gradient that needs to be sent 
        backward to the previous tensors (or operations) in the computational graph. It's a temporary storage to hold 
        gradients until this tensor has received all expected gradients, after which it will send them backward.

        This method ensures that the tensor can correctly participate in multiple operations by summing up gradients 
        from each operation it is involved in.
        """
        # Accumulate the gradient for the current tensor
        if self.grad is None:
            self.grad = incoming_gradient
        else:
            self.grad += incoming_gradient

        # Accumulate the gradient to be propagated backward
        if self._accumulated_gradient_to_propagate is None:
            self._accumulated_gradient_to_propagate = incoming_gradient
        else:
            self._accumulated_gradient_to_propagate += incoming_gradient

    def _propagate_gradient(self) -> None:
        """
        Propagate the accumulated gradient backward through the computational graph.

        This method sends the accumulated gradient to the `GradientFunction` that was responsible for creating this 
        tensor.

        The gradient is only propagated if the following conditions are met:
        1. The tensor has an associated `GradientFunction` (i.e., it is not a leaf node in the computational graph).
        2. The tensor has received all expected gradients (`_pending_gradients_count` is zero). This ensures that the 
        tensor waits until it has accumulated all gradients from the operations it was involved in.

        After propagating the gradient, the accumulated gradient used for propagation is reset to `None`, preparing this 
        tensor for potential future backward passes.
        """
        if self.grad_fn is not None and self._pending_gradients_count == 0:
            # Propagate the accumulated gradient backward to the GradientFunction that creates this tensor
            self.grad_fn.backward(self._accumulated_gradient_to_propagate)

            # Reset the accumulated gradient after propagation
            self._accumulated_gradient_to_propagate = None

    def _propagate_reference_signal(self) -> None:
        """
        Propagate a reference signal through the computational graph to prepare for gradient backpropagation.

        This method is a preparatory step in the backward pass. Its primary function is to count the number of 
        operations each tensor in the graph is involved in. This count is crucial because it determines when a tensor is 
        ready to propagate its accumulated gradient backward. A tensor will only propagate its gradient back once it has 
        received all expected gradients from the operations it participated in. 

        By tracking this count, the method minimizes unnecessary traversals of the graph, ensuring that each tensor 
        waits until it has all its required gradient contributions before participating in the backward pass. This 
        optimization is significant in complex graphs where there are millions of tensors and operations.

        The method operates in two modes:
            - Initial call (when `_pending_gradients_count` is zero): This is typically the case when the backward pass 
            is initiated. The method will propagate the reference signal to the `GradientFunction` that created it, 
            which in turn propagates the signal to its input tensors.

            - Subsequent calls (when `_pending_gradients_count` is greater than zero): In this case, the tensor has 
            already received a part of its expected gradient. The method decrements the `_pending_gradients_count` 
            count, indicating that one less gradient is now expected.
        """
        # Start or continue the process of propagating the reference signal
        if self._pending_gradients_count == 0 and self.grad_fn is not None:
            # If this is the initial call, propagate the signal to the gradient function
            self.grad_fn._propagate_reference_signal()
        elif self._pending_gradients_count > 0:
            # If this is a subsequent call, decrement the backward reference count
            self._pending_gradients_count -= 1

    def _increment_pending_gradients(self) -> None:
        """
        Increment the count of pending gradients for this tensor and propagate this increment up the graph.

        This method is invoked when a tensor is identified as part of an operation in the computational graph that 
        contributes to the final output. By incrementing the `_pending_gradients_count` count, the tensor acknowledges 
        that it is expecting to receive a gradient from this operation during the backward pass.

        The incremented count serves two purposes:
        1. It keeps track of how many operations involve this tensor, indicating how many gradients this tensor should 
        expect to receive in the backward pass.
        2. It signals the tensor to wait until all its expected gradients are received before propagating its own gradient 
        backward, thereby ensuring the correct accumulation of gradients.

        After incrementing the reference count, this method propagates the reference signal further up the graph. This 
        propagation continues until it reaches tensors that are not part of any further operations, effectively preparing 
        the entire graph for efficient gradient propagation.

        If the tensor is a result of an operation (i.e., it has a gradient function), this method also triggers the 
        reference signal propagation in the gradient function, ensuring that all tensors upstream in the graph are 
        similarly prepared.
        """
        # Increment the count of pending gradients
        self._pending_gradients_count += 1

        # Propagate the count increment signal up to the gradient function, if it exists
        if self.grad_fn is not None:
            self.grad_fn._propagate_reference_signal()

    def zero_grad(self) -> None:
        # Reset the gradient of the tensor
        self.grad = None


class GradientFunction():

    def __init__(self, operation: mlops.Operation = None, inputs: list[Tensor] = []) -> None:
        self.operation = operation
        self.inputs = inputs

        self._reference_signal_propagated = False

    def _propagate_reference_signal(self) -> None:
        """
        Propagate a reference signal through the inputs of this operation.

        This method increases the pending gradient count of each input tensor. It ensures that each tensor knows how 
        many gradients it should expect, which is crucial for the correct accumulation and backpropagation of gradients.

        The propagation occurs only once to prevent multiple counts for the same operation.
        """
        if not self._reference_signal_propagated:
            self._reference_signal_propagated = True
            for input_tensor in self.inputs:
                input_tensor._increment_pending_gradients()

    def backward(self, incoming_gradient: Buffer) -> None:
        """
        Backpropagate the gradient through the operation.

        This method computes the gradients for the operation and propagates them backward to each input tensor. The 
        backward propagation continues recursively through the graph until reaching the leaf tensors.
        """
        if self.operation:
            # Computes the gradient of the operation
            gradients = self.operation.backward(incoming_gradient)

            # We expect a tuple with the gradients but some operations have only one so we wrap the gradient in a tuple
            if not isinstance(gradients, tuple):
                gradients = (gradients,)

            # Propagate the gradient of the operation to its input tensors
            for input_tensor, grad in zip(self.inputs, gradients):
                input_tensor.backward(grad)

            # Reset the reference signal flag for potential reuse of this graph
            self._reference_signal_propagated = False

    def __str__(self) -> str:
        inputs_str = ", ".join(f"<Tensor {hex(id(tensor))}>" for tensor in self.inputs)
        return f"<GradientFunction {self.operation} with inputs [{inputs_str}]>"


def apply_op(operation_cls: mlops.Operation, *tensors: Tensor, **kwargs) -> Tensor:
    # The output tensor requires gradient if any of the input tensors also require it
    needs_input_grad = [tensor.requires_grad for tensor in tensors]
    if any(needs_input_grad):
        requires_grad = True
    elif None in needs_input_grad:
        requires_grad = None
    else:
        requires_grad = False

    # We need an object of the operation to make the forward pass
    operation_object = operation_cls(needs_input_grad)

    # Do the forward pass with the buffers of the tensors
    buffer = operation_object.forward(*[tensor.buffer for tensor in tensors], **kwargs)

    # Save the buffer in a new tensor
    output_tensor = Tensor(buffer, requires_grad=requires_grad)

    # If the output tensor requires a gradient, set up the gradient function
    if output_tensor.requires_grad:
        output_tensor.grad_fn = GradientFunction(operation=operation_object, inputs=tensors)

    return output_tensor
