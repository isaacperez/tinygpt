from __future__ import annotations
from typing import Any, Union, Iterator
from collections import deque
from enum import Enum, auto
import itertools
import random
import math

from tinygpt.utils import DType


class Buffer():

    class Op(Enum):
        # Enum for supported operations
        ADD = auto()
        SUB = auto()
        NEG = auto()
        MUL = auto()
        DIV = auto()
        POW = auto()
        MAX = auto()
        LOG = auto()
        EXP = auto()
        LT = auto()
        LE = auto()
        EQ = auto()
        NE = auto()
        GT = auto()
        GE = auto()
        UNIFORM = auto()
        NORMAL = auto()
        ZEROS = auto()
        ONES = auto()

    def __init__(self, input_data: Any, dtype: DType = None) -> None:
        # Initialize buffer based on the type of input data
        if not isinstance(dtype, DType) and dtype is not None:
            raise ValueError(f"Expecting type DType or None for dtype but found {type(dtype)}")

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

    def _get_buffer_str(self, max_display_elements: int = 2) -> str:
        # Generate a string representation of the buffer's data respecting its shape
        def recurser(index):
            # Recursive function to format the buffer data according to its multi-dimensional shape
            if len(index) == self.ndim:
                # Base case: If the current index is at the deepest dimension, return the element at that index
                return str(self._get(index))
            else:
                elements = []
                size = self.shape[len(index)]

                # Add elements from the beginning of the current dimension
                for i in range(min(size, max_display_elements)):
                    elements.append(recurser(index + (i,)))

                # Insert '...' to indicate skipped elements if the dimension size is large
                if size > 2 * max_display_elements:
                    elements.append('...')

                # Add elements from the end of the current dimension
                for i in range(max(size - max_display_elements, max_display_elements), size):
                    elements.append(recurser(index + (i,)))

                return "[" + ", ".join(elements) + "]"

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

    @staticmethod
    def _create_buffer_from_data(data: list, shape: tuple, stride: tuple, offset: int) -> Buffer:
        # Validate the buffer's data, shape, stride, and offset
        Buffer._validate_data_types(data, shape, stride, offset)
        Buffer._validate_data_values(data, shape, stride, offset)

        # Create a Buffer to store the new data
        new_buffer = Buffer([])

        # Assign validated values
        new_buffer.data = data
        new_buffer.shape = shape
        new_buffer.stride = stride
        new_buffer.offset = offset
        new_buffer.ndim = len(shape)
        new_buffer.dtype = Buffer._deduce_dtype(data)
        new_buffer.numel = Buffer._numel(shape)

        return new_buffer

    @staticmethod
    def _validate_data_types(data: list, shape: tuple, stride: tuple, offset: int) -> None:
        # Validate the types of inputs for _create_buffer_from_data
        if not isinstance(data, list):
            raise TypeError(f"Expected data to be a list, found {type(data)}")
        if not isinstance(shape, tuple):
            raise TypeError(f"Expected shape to be a tuple, found {type(shape)}")
        if not all(isinstance(value, int) for value in shape):
            raise TypeError(f"Expected shape values to be a integers, found {shape}")
        if not isinstance(stride, tuple):
            raise TypeError(f"Expected stride to be a tuple, found {type(stride)}")
        if not all(isinstance(value, int) for value in shape):
            raise TypeError(f"Expected stride values to be a integers, found {stride}")
        if not isinstance(offset, int):
            raise TypeError(f"Expected offset to be an int, found {type(offset)}")

    @staticmethod
    def _validate_data_values(data: list, shape: tuple, stride: tuple, offset: int) -> None:
        # Validate the values of inputs for _create_buffer_from_data
        data_size = len(data)
        if len(shape) != len(stride):
            raise ValueError("Length of shape and stride must match")
        if not 0 <= offset < data_size:
            raise ValueError("Offset must be within the range of data elements")

        Buffer._validate_stride_and_shape(data_size, shape, stride, offset)

    @staticmethod
    def _validate_stride_and_shape(data_size: int, shape: tuple, stride: tuple, offset: int) -> None:
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

    def __iter__(self) -> Iterator[Union[float, int, bool]]:
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

    def _execute(self, op: Op, other: Union[Buffer, int, float]) -> Buffer:
        # Execute a specified operation between two buffers
        # Ensures that both buffers are of the same dtype and shape before performing the operation
        self._validate_input_buffer(op, other)

        # Create a generator that yields the scalar value 'numel' times if 'other' is a scalar
        if not isinstance(other, Buffer) and op != self.Op.POW:
            def scalar_generator(value, count):
                for _ in range(count):
                    yield value

            other = scalar_generator(other, self.numel)

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
        elif op == self.Op.POW:
            data = [first_element ** other for first_element in self]
        elif op == self.Op.LOG:
            data = [math.log(element) for element in self]
        elif op == self.Op.EXP:
            data = [math.exp(element) for element in self]
        elif op == self.Op.MAX:
            data = [max(first_element, second_element) for first_element, second_element in zip(self, other)]
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

        # Assign the data and its metadata to the new buffer
        return Buffer._create_buffer_from_data(
            data=data, shape=self.shape, stride=Buffer._calculate_stride(self.shape), offset=0
        )

    def _validate_input_buffer(self, op: Op, other: Union[Buffer, int, float]) -> None:
        # Validate compatibility of two buffers for an operation
        if not isinstance(op, self.Op):
            raise TypeError(f"op argument is not an Op. Found {type(op)}")

        # For the POW operation, we only accept int or float as an exponent
        if op == self.Op.POW:
            if not isinstance(other, (int, float)):
                raise TypeError("Only supporting int/float powers for now, found {type(other)}")

        else:
            # We only accept Buffer, int or float
            if not isinstance(other, (int, float, Buffer)):
                raise TypeError(f"Expected a Buffer object, float or int for operation, found {type(other)}")

            # If it is a buffer, it must have the same type and shape as the current buffer
            if isinstance(other, Buffer):
                if self.dtype != other.dtype:
                    raise ValueError(f"DType mismatch. Found {self.dtype} and {other.dtype}")

                if self.shape != other.shape:
                    raise RuntimeError(f"Shape mismatch. Found {self.shape} and {other.shape}")

    def __add__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.ADD, other)

    def __radd__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self + other

    def __sub__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.SUB, other)

    def __rsub__(self, other: Union[Buffer, int, float]) -> Buffer:
        return (-self) + other

    def __neg__(self) -> Buffer:
        return self._execute(self.Op.NEG, self)

    def __mul__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.MUL, other)

    def __rmul__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self * other

    def __truediv__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.DIV, other)

    def __rtruediv__(self, other: Union[Buffer, int, float]) -> Buffer:
        return other * self ** -1

    def __pow__(self, exponent: Union[int, float]) -> Buffer:
        return self._execute(self.Op.POW, exponent)

    def log(self) -> Buffer:
        return self._execute(self.Op.LOG, self)

    def exp(self) -> Buffer:
        return self._execute(self.Op.EXP, self)

    def __lt__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.LT, other)

    def __le__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.LE, other)

    def __eq__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.EQ, other)

    def __ne__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.NE, other)

    def __gt__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.GT, other)

    def __ge__(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.GE, other)

    def maximum(self, other: Union[Buffer, int, float]) -> Buffer:
        return self._execute(self.Op.MAX, other)

    def float(self) -> Buffer:
        # Create a Buffer with the same data and float dtype
        if self.dtype == DType.float32:
            return self
        elif self.ndim == 0:
            return Buffer(DType.float32.cast(self.data[0]))
        else:
            return Buffer._create_buffer_from_data(
                data=[DType.float32.cast(element) for element in self._get_contiguous_data()],
                shape=self.shape,
                stride=self._calculate_stride(self.shape),
                offset=0
            )

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
        reshaped_buffer = Buffer._create_buffer_from_data(data, new_shape, self._calculate_stride(new_shape), offset)

        return reshaped_buffer

    def expand(self, new_shape: tuple) -> Buffer:
        # Expand the buffer to a new shape. Dimensions with size 1 can be expanded to larger sizes
        if not isinstance(new_shape, tuple):
            raise TypeError(f"Expecting type tuple for new_shape but found type {type(new_shape)}")

        # Check if the number of dimensions is compatible
        if len(new_shape) != self.ndim:
            raise ValueError("Expanded shape must have as many dimensions as the original")

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
        expanded_buffer = Buffer._create_buffer_from_data(data, new_shape, new_stride, offset)

        return expanded_buffer

    def _generate_indexes(self) -> Iterator[tuple]:
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
        new_buffer = Buffer._create_buffer_from_data(
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
        elif op == self.Op.MAX:
            for index in new_buffer._generate_indexes():
                reduced_value = self._calculate_reduced_value_with_max(buffer, index, axis)
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

    def _calculate_reduced_value_with_max(self, buffer: Buffer, index: tuple, axis: int) -> float:
        # Calculate the reduced value for a given index by using the max operation
        old_index = list(index)
        old_index[axis] = 0
        reduced_value = buffer._get(tuple(old_index))
        for i in range(1, buffer.shape[axis]):
            old_index[axis] = i
            reduced_value = max(reduced_value, buffer._get(tuple(old_index)))

        return reduced_value

    def sum(self, axes: tuple) -> Buffer:
        return self._reduce(self.Op.ADD, axes)

    def max(self, axes: tuple) -> Buffer:
        return self._reduce(self.Op.MAX, axes)

    @staticmethod
    def concatenate(buffers: list[Buffer], axis: int = 0) -> Buffer:
        if not buffers:
            raise ValueError("Need at least one buffer to concatenate")
        
        # Check that all elements in buffers are instances of Buffer
        if any(not isinstance(buffer, Buffer) for buffer in buffers):
            raise TypeError("All elements must be of type Buffer")
        
        # Check if all buffers have the same number of dimensions
        ndim = buffers[0].ndim
        if any(buffer.ndim != ndim for buffer in buffers):
            raise ValueError("All buffers must have the same number of dimensions")

        # Normalize axis to handle negative values
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Axis out of bounds for concatenation: {axis}")
        
        # Check that all buffers have the same shape except along the concatenation axis
        shape = list(buffers[0].shape)
        for buffer in buffers[1:]:
            for dim in range(ndim):
                if dim != axis and shape[dim] != buffer.shape[dim]:
                    raise ValueError("All buffers must have the same shape, except along the concatenation axis")

        # Check that all buffers have the same dtype
        dtype = buffers[0].dtype
        for buffer in buffers[1:]:
            if buffer.dtype != dtype:
                raise ValueError("All buffers must have the same dtype")

        # Calculate the new shape after concatenation
        new_shape = list(shape)
        new_shape[axis] = sum(buffer.shape[axis] for buffer in buffers)

        # Initialize a zero buffer with the new shape
        result_buffer = Buffer.zeros(tuple(new_shape), dtype)

        # Place each buffer's values into the appropriate position in the result buffer
        start_idx = 0
        for buffer in buffers:
            end_idx = start_idx + buffer.shape[axis]
            slices = [slice(None)] * ndim
            slices[axis] = slice(start_idx, end_idx)
            result_buffer[tuple(slices)] = buffer
            start_idx = end_idx

        return result_buffer

    def tril(self, diagonal: int = 0) -> Buffer:
        # Ensure the buffer is at least 2-dimensional
        if self.ndim < 2:
            raise ValueError("tril requires the buffer to be at least 2-dimensional")

        # Copy the data of the tensor because we are going to modify it
        new_data = [element for element in self]

        # Create a new buffer with the same shape and dtype
        new_buffer = Buffer._create_buffer_from_data(
            data=new_data, shape=self.shape, stride=Buffer._calculate_stride(self.shape), offset=0
        )

        if self.ndim == 2:
            self._apply_tril(new_buffer, diagonal)
        else:
            # Iterate over all 2D slices for higher dimensional buffers
            for indices in itertools.product(*(range(dim) for dim in self.shape[:-2])):
                # Convert indices to a full index with slice(None) for the last two dimensions
                full_indices = tuple(indices) + (slice(None), slice(None))
                self._apply_tril(new_buffer[full_indices], diagonal)

        return new_buffer

    def _apply_tril(self, buffer: Buffer, diagonal: int) -> None:
        # Apply tril to a 2D buffer
        for i in range(buffer.shape[-2]):
            for j in range(buffer.shape[-1]):
                if j - i > diagonal:
                    buffer._set((i, j), 0)

    @staticmethod
    def _init(init_op: Buffer.Op, shape: tuple, dtype: DType = DType.float32) -> Buffer:
        # Initialize a new Buffer using a specified operation and shape

        # Validate the type of init_op and shape
        if not isinstance(init_op, Buffer.Op):
            raise TypeError(f"init_op argument is not an Op. Found {type(init_op)}")

        if not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple")

        # Validate that each dimension in shape is positive
        if any(value < 1 for value in shape):
            raise ValueError(f"One or more values in shape is not positive: {shape}")

        # Calculate the total number of elements
        numel = Buffer._numel(shape)

        # Initialize buffer data based on the specified operation
        if init_op == Buffer.Op.UNIFORM:
            # Generate random values uniformly distributed between 0 and 1
            data = [dtype.cast(random.uniform(0.0, 1.0)) for _ in range(numel)]
        elif init_op == Buffer.Op.NORMAL:
            # Generate random values using Normal distribution
            data = [dtype.cast(random.gauss(mu=0, sigma=1)) for _ in range(numel)]
        elif init_op == Buffer.Op.ZEROS:
            data = [dtype.cast(0.0) for _ in range(numel)]
        elif init_op == Buffer.Op.ONES:
            data = [dtype.cast(1.0) for _ in range(numel)]
        else:
            raise RuntimeError(f"Initialization operation {init_op.value} not implemented")

        # Return a new Buffer with the generated data and the specified shape
        return Buffer(data).reshape(shape)

    @staticmethod
    def uniform(shape: tuple, dtype: DType = DType.float32) -> Buffer:
        # Create a Float Buffer with data initialized uniformly between 0 and 1
        return Buffer._init(Buffer.Op.UNIFORM, shape, dtype)

    @staticmethod
    def normal(shape: tuple, dtype: DType = DType.float32) -> Buffer:
        # Create a Float Buffer with data initialized randomly by a normal distribution
        return Buffer._init(Buffer.Op.NORMAL, shape, dtype)
    
    @staticmethod
    def zeros(shape: tuple, dtype: DType = DType.float32) -> Buffer:
        # Create a Float Buffer with data initialized with 0s
        return Buffer._init(Buffer.Op.ZEROS, shape, dtype)

    @staticmethod
    def ones(shape: tuple, dtype: DType = DType.float32) -> Buffer:
        # Create a Float Buffer with data initialized with 0s
        return Buffer._init(Buffer.Op.ONES, shape, dtype)

    def _generate_indices_with_custom_order(self, order: tuple) -> Iterator[tuple]:
        # Generate indices for the buffer by incrementing dimensions in a specified custom order

        if len(order) != self.ndim:
            raise ValueError("Length of order must match the number of dimensions of the buffer")

        # Initialize the indices for each dimension
        indices = [0] * self.ndim

        # Iterate over the elements as per the custom order
        while True:
            yield tuple(indices)

            # Increment the indices in the specified order
            for dim in order:
                if indices[dim] < self.shape[dim] - 1:
                    indices[dim] += 1
                    break
                else:
                    indices[dim] = 0
            else:
                # If all indices have rolled over, the iteration is complete
                return

    def permute(self, dims: tuple[int, ...]) -> Buffer:
        # Rearrange the dimensions of the buffer according to a specified order

        if not isinstance(dims, tuple):
            raise TypeError("dims must be a tuple")

        # Check if the number of dimensions provided matches the buffer's dimensions
        if len(dims) != self.ndim:
            raise ValueError("Number of dimensions provided doesn't match the buffer's dimensions")

        # Check if the provided dimensions form a valid permutation of the current dimensions
        if set(dims) != set(range(self.ndim)):
            raise ValueError("dims don't form a valid permutation")

        # Calculate the new shape by rearranging the current shape according to the provided dimensions
        new_shape = tuple(self.shape[dim] for dim in dims)

        # Create the new Buffer
        if self.is_contiguous():
            # Calculate the new stride for the contiguous data
            new_stride = tuple(self.stride[dim] for dim in dims)
            # For contiguous data, avoid data movement
            return Buffer._create_buffer_from_data(self.data, new_shape, new_stride, self.offset)

        else:
            # Reorder the data according to the new dimension order
            reordered_data = [self._get(index) for index in self._generate_indices_with_custom_order(dims[::-1])]

            # Calculate the new stride for the reordered (now contiguous) data
            new_stride = self._calculate_stride(new_shape)

            return Buffer._create_buffer_from_data(reordered_data, new_shape, new_stride, 0)

    def to_python(self) -> Union[float, int, bool, list]:
        # Convert the buffer's data to a Python scalar or nested list.
        # Handle scalar buffer
        if self.ndim == 0:
            return self._get((0,))

        # Handle multi-dimensional buffer
        return self._convert_to_nested_list(self.shape, [])

    def _convert_to_nested_list(self, shape, index):
        # Helper method to recursively build a nested list from the buffer's data.
        # Base case: If the index length equals the number of dimensions, retrieve the value
        if len(index) == self.ndim:
            return self._get(tuple(index))

        # Recursive case: Build a list for the current dimension
        nested_list = []
        for i in range(shape[len(index)]):
            nested_list.append(self._convert_to_nested_list(shape, index + [i]))

        return nested_list
    
    def __getitem__(self, index: Union[int, slice, tuple]) -> Buffer:
        # Normalize the index to a tuple to handle both single and multi-dimensional indexing
        if not isinstance(index, tuple):
            index = (index,)
        
        if len(index) > self.ndim:
            raise IndexError(f"Too many indices for {self.ndim}-dim Buffer")

        # Extend the index with slices to match the number of dimensions
        index = index + (slice(None),) * (self.ndim - len(index))

        # Convert slices to ranges and handle integer indices
        new_shape = []
        new_stride = []
        new_offset = self.offset
        full_index = []
        for idx, (dim, stride) in zip(index, zip(self.shape, self.stride)):
            if isinstance(idx, int):
                # Adjust for negative index
                if idx < 0: 
                    idx += dim
                if idx < 0 or idx >= dim:
                    raise IndexError("Index out of range")
                
                new_offset += idx * stride  
                full_index.append([idx])

            elif isinstance(idx, slice):
                # Convert the slice to a range
                start, stop, step = idx.indices(dim)

                if step < 0:
                    raise ValueError("step must be greater than zero")
                
                slice_range = range(start, stop, step)

                new_shape.append(len(slice_range))
                new_stride.append(stride * step)
                new_offset += start * stride
                full_index.append(slice_range)

            else:
                raise TypeError("Invalid index type")

        # Return an empty Buffer if any dimension has length 0
        if any(dim == 0 for dim in new_shape):
            return Buffer([], dtype=self.dtype)
        
        if self.is_contiguous() or not new_shape:
            # For contiguous data or single element, avoid data movement
            return Buffer._create_buffer_from_data(self.data, tuple(new_shape), tuple(new_stride), new_offset)
        
        else:
            # Generate new data for the sliced Buffer using _get method
            new_data = []
            for element_indices in itertools.product(*full_index):
                new_data.append(self._get(tuple(element_indices))) 

            # Calculate new stride for the new contiguous buffer
            new_stride = self._calculate_stride(new_shape)

            return Buffer._create_buffer_from_data(new_data, tuple(new_shape), new_stride, 0)
        
    def __setitem__(self, index: Union[int, slice, tuple], value: Union[Buffer, float, int, bool]) -> None:
        # Normalize the index to a tuple to handle both single and multi-dimensional indexing
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) > self.ndim:
            raise IndexError(f"Too many indices for {self.ndim}-dim Buffer")

        # Extend the index with slices to match the number of dimensions
        index = index + (slice(None),) * (self.ndim - len(index))

        # Convert slices to ranges and handle integer indices
        full_index = []
        for idx, dim  in zip(index, self.shape):
            if isinstance(idx, int):
                # Adjust for negative index
                if idx < 0:
                    idx += dim
                if idx < 0 or idx >= dim:
                    raise IndexError("Index out of range")
                full_index.append([idx])

            elif isinstance(idx, slice):
                # Convert the slice to a range
                start, stop, step = idx.indices(dim)

                if step < 0:
                    raise ValueError("step must be greater than zero")
                
                slice_range = range(start, stop, step)
                full_index.append(slice_range)

            else:
                raise TypeError("Invalid index type")

        # Do nothing if any of the ranges are empty
        if any(len(r) == 0 for r in full_index):
            return 

        # Update the buffer with the value
        if isinstance(value, Buffer):
            # Both the current Buffer and the value Buffer must have the same shape or be broadcastable
            index_shape = tuple(len(r) for r in full_index)
            if [idx for idx in index_shape if idx != 1] != [idx for idx in value.shape if idx != 1]:
                raise ValueError(f"Shape mismatch: {index_shape} and {value.shape}")
            
            # Set buffer values using the provided buffer's data
            for element_indices, val in zip(itertools.product(*full_index), value):
                self._set(tuple(element_indices), val)

        elif isinstance(value, (bool, int, float)):
            # Set single value to all selected positions
            for element_indices in itertools.product(*full_index):
                self._set(tuple(element_indices), value)

        else:
            raise ValueError("value is not a Buffer or a single value (bool, int or float)")