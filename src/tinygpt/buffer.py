from __future__ import annotations
from enum import Enum, auto
from collections import deque
from typing import Any, Union

from tinygpt.utils import DType


class Buffer():

    class Op(Enum):
        # Enum for supported operations
        ADD = auto()
        SUB = auto()
        NEG = auto()
        MUL = auto()
        DIV = auto()
        LT = auto()
        LE = auto()
        EQ = auto()
        NE = auto()
        GT = auto()
        GE = auto()

    def __init__(self, input_data: Any, dtype: DType = None) -> None:
        # Initialize buffer based on the type of input data
        if isinstance(input_data, Buffer):
            self._copy_from_existing_buffer(input_data, dtype)
        else:
            self._process_input_data(input_data, dtype)

    def __repr__(self) -> str:
        return f"<Buffer {hex(id(self))}: {self._get_buffer_str()}, shape={self.shape}, dtype={self.dtype}>"

    def __str__(self) -> str:
        return f"{self._get_buffer_str()}"

    def _copy_from_existing_buffer(self, input_buffer: Buffer, dtype: DType | None) -> None:
        # Copying from an existing buffer, ensuring dtype compatibility
        if dtype is not None and dtype != input_buffer.dtype:
            raise RuntimeError("dtype doesn't match, and casting isn't supported")

        self.data = input_buffer.data
        self.offset = input_buffer.offset
        self.ndim = input_buffer.ndim
        self.stride = input_buffer.stride
        self.shape = input_buffer.shape
        self.dtype = input_buffer.dtype
        self.numel = input_buffer.numel

    def _process_input_data(self, input_data: Any, dtype: DType | None) -> None:
        self.offset = 0
        self.dtype = dtype or self._deduce_dtype(input_data)

        if isinstance(input_data, (list, tuple)):
            self._process_array_input(input_data)
        elif isinstance(input_data, (float, int, bool)):
            self._process_scalar_input(input_data)
        else:
            raise RuntimeError(f"Could not infer dtype of type {type(input_data)}")

    def _process_array_input(self, input_data: Any) -> None:
        # Process array input data
        self.data, self.shape = self._extract_flat_array_and_shape(input_data, self.dtype)
        self.stride = self._calculate_stride(self.shape)
        self.ndim = len(self.shape)
        self.numel = self._numel(self.shape)

    def _process_scalar_input(self, input_data: Any) -> None:
        # Process scalar input data
        self.data = [self.dtype.cast(input_data)]
        self.shape = ()
        self.stride = ()
        self.ndim = 0
        self.numel = self._numel(self.shape)

    def _get_buffer_str(self) -> str:
        # Generate a string representation of the buffer's data respecting its shape
        def recurser(index):
            # Recursive function to format the buffer data according to its multi-dimensional shape
            if len(index) == self.ndim:
                return str(self._get(index))
            else:
                return "[" + ", ".join(recurser(index + (i,)) for i in range(self.shape[len(index)])) + "]"

        return recurser(())

    def _extract_flat_array_and_shape(self, input_data: Any, dtype: DType) -> tuple[list, list]:
        # Converts nested lists or tuples to a flat array while extracting the shape
        # The size and type of each element in each dimension must be consistent
        flat_array = []
        size_by_dim = {}
        type_by_dim = {}
        queue = deque([(input_data, 1)])

        while queue:
            current_element, dim = queue.popleft()

            # Check type consistency
            expected_type = type_by_dim.setdefault(dim, type(current_element))
            if not isinstance(current_element, expected_type):
                raise TypeError(f"Type mismatch at dim {dim}: expected {expected_type}, got {type(current_element)}")

            if isinstance(current_element, (list, tuple)):
                # Check size consistency
                expected_size = size_by_dim.setdefault(dim, len(current_element))
                if len(current_element) != expected_size:
                    raise ValueError(
                        f"Inconsistent size at dim {dim}: expected {expected_size}, got {len(current_element)}"
                    )

                # Queue child elements for further processing
                queue.extend((element, dim + 1) for element in current_element)

            else:
                # Add element to the flat array
                flat_array.append(dtype.cast(current_element))

        shape = tuple(size_by_dim.values())

        return flat_array, shape

    @staticmethod
    def _calculate_stride(shape: tuple) -> tuple:
        # Calculate the stride needed to traverse the buffer given its shape
        # Stride is the number of elements to skip in each dimension to move to the next element
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]

        return tuple(stride)

    @staticmethod
    def _deduce_dtype(data: Any) -> DType:
        # Deduce the data type (DType) from the input data
        # If data is a list or tuple, find the first non-list element to deduce its type
        # If there is non-list element, the default type is DType.float32
        first_element = data
        while isinstance(first_element, (list, tuple)) and first_element:
            first_element = first_element[0]

        return DType.deduce_dtype(first_element) if not isinstance(first_element, (list, tuple)) else DType.float32

    def _set_data(self, data: list, shape: tuple, stride: tuple, offset: int) -> None:
        # Set and validate the buffer's data, shape, stride, and offset
        self._validate_set_data_input_types(data, shape, stride, offset)
        self._validate_set_data_values(data, shape, stride, offset)

        # Assign validated values
        self.data = data
        self.shape = shape
        self.stride = stride
        self.offset = offset
        self.ndim = len(shape)
        self.dtype = self._deduce_dtype(data)
        self.numel = self._numel(shape)

    def _validate_set_data_input_types(self, data: list, shape: tuple, stride: tuple, offset: int) -> None:
        # Validate the types of inputs for _set_data
        if not isinstance(data, list):
            raise TypeError(f"Expected data to be a list, found {type(data)}")
        if not isinstance(shape, tuple):
            raise TypeError(f"Expected shape to be a tuple, found {type(shape)}")
        if not isinstance(stride, tuple):
            raise TypeError(f"Expected stride to be a tuple, found {type(stride)}")
        if not isinstance(offset, int):
            raise TypeError(f"Expected offset to be an int, found {type(offset)}")

    def _validate_set_data_values(self, data: list, shape: tuple, stride: tuple, offset: int) -> None:
        # Validate the values of inputs for _set_data
        data_size = len(data)
        if len(shape) != len(stride):
            raise ValueError("Length of shape and stride must match")
        if not 0 <= offset < data_size:
            raise ValueError("Offset must be within the range of data elements")

        self._validate_stride_and_shape(data_size, shape, stride, offset)

    def _validate_stride_and_shape(self, data_size: int, shape: tuple, stride: tuple, offset: int) -> None:
        # Validate that the stride and shape correspond correctly to the data size
        max_flat_index = offset
        for dim_size, dim_stride in zip(shape, stride):
            if dim_size <= 0:
                raise ValueError("Dimensions in shape must be positive")
            if dim_stride < 0:
                raise ValueError("Stride values must be non-negative")

            max_flat_index += (dim_size - 1) * dim_stride

        if max_flat_index >= data_size:
            raise ValueError("Stride and shape are not compatible with data size")

    def _index_to_flat_index(self, index: tuple) -> int:
        # From a multidimensional index tuple to its corresponding index in the flat array
        return self.offset + sum([idx * stride for idx, stride in zip(index, self.stride)])

    @staticmethod
    def _numel(shape: tuple) -> int:
        # Calculate the total number of elements in a buffer with the specified shape
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        return total_elements

    def __iter__(self) -> Union[float, int, bool]:
        # Iterator to enable looping over the buffer elements
        # This accounts for the multi-dimensional nature of the buffer
        if self.ndim == 0:
            # Yield the single element if the buffer is a scalar
            yield self._get((0,))
        else:
            # Iterate over each element according to the buffer's shape
            indices = [0] * self.ndim
            for _ in range(self.numel):
                yield self._get(tuple(indices))
                self._update_indices(indices)

    def _update_indices(self, indices: list[int]) -> None:
        # Helper method to update the indices for the iterator
        for i in range(self.ndim - 1, -1, -1):
            if indices[i] < self.shape[i] - 1:
                indices[i] += 1
                break
            else:
                indices[i] = 0

    def _set(self, index: tuple, value: Any) -> None:
        # Internal method to set a value at a specific index
        # This method bypasses the standard Python list assignment to handle multi-dimensional indexing
        self.data[self._index_to_flat_index(index)] = self.dtype.cast(value)

    def _get(self, index: tuple) -> Union[float, int, bool]:
        # Internal method to get a value at a specific index
        # Converts a multi-dimensional index to a flat index and retrieves the value
        return self.data[self._index_to_flat_index(index)]

    def is_contiguous(self) -> bool:
        # Check if the buffer's data is contiguous in memory
        # A buffer is contiguous if each element is next to its multi-dimensional neighbor in memory
        expected_stride = 1
        for dim, stride in reversed(list(zip(self.shape, self.stride))):
            if stride != expected_stride:
                return False
            expected_stride *= dim

        return True

    def _get_contiguous_data(self) -> list:
        # Create a contiguous copy of the buffer data
        # Useful for operations that require contiguous data layout
        return [element for element in self]

    @staticmethod
    def _broadcastable(first_buffer: Buffer, second_buffer: Buffer) -> bool:
        # Check if two buffers are broadcastable to each other
        # Broadcasting allows for operations on arrays of different shapes
        for dim_first, dim_second in zip(reversed(first_buffer.shape), reversed(second_buffer.shape)):
            if dim_first != 1 and dim_first != dim_second:
                return False

        return True

    def _execute(self, op: Op, other: Buffer) -> Buffer:
        # Execute a specified operation between two buffers
        # Ensures that both buffers are of the same dtype and shape before performing the operation
        self._validate_input_buffer(op, other)

        # Perform the operation element-wise
        if op == self.Op.ADD:
            data = [first_element + second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.SUB:
            data = [first_element - second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.NEG:
            data = [-first_element for first_element in self]
        elif op == self.Op.MUL:
            data = [first_element * second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.DIV:
            data = [first_element / second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.LT:
            data = [first_element < second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.LE:
            data = [first_element <= second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.EQ:
            data = [first_element == second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.NE:
            data = [first_element != second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.GT:
            data = [first_element > second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.GE:
            data = [first_element >= second_element for first_element, second_element in zip(self, other)]
        else:
            raise RuntimeError(f"Operation {op.value} not implemented")

        # Create a buffer to store the result
        result = Buffer([])

        # Assign the data and its metadata to the new buffer
        # The shape and stride of the result are the same as the input buffers
        result._set_data(data=data, shape=self.shape, stride=Buffer._calculate_stride(self.shape), offset=0)

        return result

    def _validate_input_buffer(self, op: Op, other: Buffer) -> None:
        # Validate compatibility of two buffers for an operation
        if not isinstance(op, self.Op):
            raise TypeError(f"op argument is not an Op. Found {type(op)}")

        if not isinstance(other, Buffer):
            raise TypeError(f"Expected a Buffer object for operation, found {type(other)}")

        if self.dtype != other.dtype:
            raise ValueError(f"DType mismatch. Found {self.dtype} and {other.dtype}")

        if self.shape != other.shape:
            raise RuntimeError(f"Shape mismatch. Found {self.shape} and {other.shape}")

    def __add__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.ADD, other)

    def __sub__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.SUB, other)

    def __neg__(self):
        return self._execute(self.Op.NEG, self)

    def __mul__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.MUL, other)

    def __truediv__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.DIV, other)

    def __lt__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.LT, other)

    def __le__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.LE, other)

    def __eq__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.EQ, other)

    def __ne__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.NE, other)

    def __gt__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.GT, other)

    def __ge__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.GE, other)

    def reshape(self, new_shape: tuple) -> Buffer:
        # Reshape the buffer to a new shape
        if not isinstance(new_shape, tuple):
            raise TypeError("new_shape must be a tuple")

        # Validate new_shape
        if any(value < 1 for value in new_shape):
            raise ValueError(f"One or more values in new_shape is not positive: {new_shape}")

        # Validate that the total number of elements remains the same
        if self.numel != Buffer._numel(new_shape):
            raise RuntimeError("Total number of elements must remain constant during reshape")

        # Use existing data if it's contiguous, otherwise create a contiguous copy
        data = self.data if self.is_contiguous() else self._get_contiguous_data()
        offset = self.offset if self.is_contiguous() else 0

        # Create a new buffer with the reshaped data
        reshaped_buffer = Buffer([])
        new_stride = self._calculate_stride(new_shape)
        reshaped_buffer._set_data(data, new_shape, new_stride, offset)

        return reshaped_buffer

    def expand(self, new_shape: tuple) -> Buffer:
        # Expand the buffer to a new shape. Dimensions with size 1 can be expanded to larger sizes
        if not isinstance(new_shape, tuple):
            raise TypeError(f"Expecting type tuple for new_shape but found type {type(new_shape)}")

        # Check if the number of dimensions is compatible
        if len(new_shape) < self.ndim:
            raise ValueError("Expanded shape must have at least as many dimensions as the original")

        # Check if sizes are compatible for expansion
        for idx, (new_size, original_size) in enumerate(zip(new_shape, self.shape)):
            if original_size != 1 and new_size != original_size:
                raise ValueError(f"Incompatible size at dimension {idx}: {original_size} cannot expand to {new_size}")

        # Determine if the buffer is contiguous
        is_contiguous = self.is_contiguous()

        # Use existing data if it's contiguous, otherwise create a contiguous copy
        data = self.data if is_contiguous else self._get_contiguous_data()
        offset = self.offset if is_contiguous else 0

        # Calculate stride for the expanded buffer
        # If the data is not contiguous, we use the stride for contiguous data
        stride_to_use = self.stride if is_contiguous else Buffer._calculate_stride(self.shape)
        new_stride = tuple(
            stride_to_use[idx] if self.shape[idx] != 1 or new_shape[idx] == 1 else 0 for idx in range(len(new_shape))
        )

        # Create a new buffer with the expanded data
        expanded_buffer = Buffer([])
        expanded_buffer._set_data(data, new_shape, new_stride, offset)

        return expanded_buffer

    def _generate_indexes(self) -> tuple:
        # Generate all possible multi-dimensional indices for the current buffer
        if self.ndim == 0:
            # Handle scalar buffer
            yield (0,)
        else:
            # Generate indices for multi-dimensional buffer
            indices = [0] * self.ndim
            for _ in range(self.numel):
                yield tuple(indices)
                self._update_indices(indices)

    def _reduce(self, op: Op, axes: tuple) -> Buffer:
        # Apply a reduction operation along specified axes
        if not isinstance(op, self.Op):
            raise TypeError(f"Expecting type Op for op but found type {type(op)}")

        # Normalize axes to a tuple
        axes = (axes,) if isinstance(axes, int) else axes
        if any(axis < 0 or axis >= self.ndim for axis in axes):
            raise ValueError(f"One or more axes are out of bounds for buffer with {self.ndim} dimensions: {axes}")

        # Perform reduction along each axis in reverse order
        reduced_buffer = self
        for axis in sorted(axes, reverse=True):
            reduced_buffer = self._reduce_along_axis(reduced_buffer, axis, op)

        return reduced_buffer

    def _reduce_along_axis(self, buffer: Buffer, axis: int, op: Op) -> Buffer:
        # Perform reduction along a specific axis

        # Calculate the new shape with the specified axis reduced to 1
        new_shape = tuple(dim if idx != axis else 1 for idx, dim in enumerate(buffer.shape))
        new_buffer = Buffer([])
        new_buffer._set_data(
            [buffer.dtype.cast(0)] * self._numel(new_shape),
            shape=new_shape,
            stride=self._calculate_stride(new_shape),
            offset=0
        )

        # Apply the operation along the axis
        if op == self.Op.ADD:
            for index in new_buffer._generate_indexes():
                reduced_value = self._calculate_reduced_value_with_sum(buffer, index, axis)
                new_buffer._set(index, reduced_value)
        else:
            raise RuntimeError(f"Operation {op.value} not implemented")

        return new_buffer

    def _calculate_reduced_value_with_sum(self, buffer: Buffer, index: tuple, axis: int) -> float:
        # Calculate the reduced value for a given index by using the sum operation
        old_index = list(index)
        reduced_value = 0
        for i in range(buffer.shape[axis]):
            old_index[axis] = i
            reduced_value += buffer._get(tuple(old_index))

        return reduced_value

    def sum(self, axes: tuple) -> Buffer:
        return self._reduce(self.Op.ADD, axes)
