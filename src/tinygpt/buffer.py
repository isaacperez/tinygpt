from __future__ import annotations
from enum import Enum, auto
from collections import deque
from typing import Any, Union

from tinygpt.utils import DType


class Buffer():

    class Op(Enum):
        SUM = auto()
        SUB = auto()
        MUL = auto()
        DIV = auto()
        LT = auto()
        LE = auto()
        EQ = auto()
        NE = auto()
        GT = auto()
        GE = auto()

    def __init__(
            self,
            input_data: Any,
            dtype: DType = None
    ) -> None:
        # Check if the input_data is already a Buffer object and copy it
        if isinstance(input_data, Buffer):

            # We don't allow casting when copying another buffer
            if dtype is not None and dtype != input_data.dtype:
                raise RuntimeError("dtype doesn't match, and casting isn't supported")

            # Copy the data
            self.data = input_data.data
            self.offset = input_data.offset
            self.ndim = input_data.ndim
            self.stride = input_data.stride
            self.shape = input_data.shape
            self.dtype = input_data.dtype

        else:
            # Copy the input data into a flat list and get some metadata to handle multidimensional operations
            self._extract_data(input_data, dtype)

    def __repr__(self) -> str:
        return f"<Buffer {self._get_buffer_str()}, shape={self.shape}, dtype={self.dtype}>"

    def __str__(self) -> str:
        return f"{self._get_buffer_str()}"

    def _get_buffer_str(self,) -> str:
        def recurser(index):
            if len(index) == self.ndim:
                return str(self._get(index))
            else:
                return "[" + ", ".join(recurser(index + (i,)) for i in range(self.shape[len(index)])) + "]"

        return recurser(())

    def _extract_flat_array_and_shape(self, input_data: Any, dtype: DType) -> (list, list):
        # The size and type of each element in each dimension must always be the same in the same dimension. We assume
        # that the first element of a dimension designates the type and size expected for the rest. While we go through
        # the data object extracting values for the flat array, we make the corresponding checks.
        flat_array = []
        size_by_dim = {}
        type_by_dim = {}
        queue = deque([(input_data, 1)])
        while len(queue) > 0:
            current_element, dim = queue.popleft()
            current_element_is_a_list = isinstance(current_element, (list, tuple))

            size_by_dim[dim] = size_by_dim.get(dim, len(current_element)) if current_element_is_a_list else -1
            type_by_dim[dim] = type_by_dim.get(dim, type(current_element))

            if current_element_is_a_list and len(current_element) != size_by_dim[dim]:
                raise ValueError(
                    f"expected sequence of length {size_by_dim[dim]} at dim {dim} (got {len(current_element)})"
                )

            elif not isinstance(current_element, type_by_dim[dim]):
                raise TypeError(f"expected type {type_by_dim[dim]} at dim {dim}, not {type(current_element)}")

            if current_element_is_a_list:
                queue.extend((element, dim + 1) for element in current_element)
            else:
                flat_array.append(dtype.cast(current_element))

        return flat_array, tuple([value for value in size_by_dim.values() if value != -1])

    @staticmethod
    def _calculate_stride(shape: tuple) -> tuple:
        # Creates a tuple with the number of elements of the flat array to be skipped in each dimension to traverse it
        ndim = len(shape)
        stride = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]

        return tuple(stride)

    @staticmethod
    def _deduce_dtype(data: Any) -> DType:
        # If it's a list, find the first element if there is one
        first_element = data
        while isinstance(first_element, (tuple, list)) and len(first_element) > 0:
            first_element = first_element[0]

        # By default, an empty buffer use DType.float32. If there is a value we deduce its DType
        if isinstance(first_element, (tuple, list)):
            return DType.float32
        else:
            return DType.deduce_dtype(first_element)

    def _extract_data(self, input_data: Any, dtype: DType | None) -> None:
        self.offset = 0

        if dtype is None:
            self.dtype = self._deduce_dtype(input_data)
        else:
            self.dtype = dtype

        if isinstance(input_data, (list, tuple)):
            self.data, self.shape = self._extract_flat_array_and_shape(input_data, self.dtype)
            self.stride = self._calculate_stride(self.shape)
            self.ndim = len(self.shape)

        elif isinstance(input_data, (float, int, bool)):
            self.data = [self.dtype.cast(input_data)]
            self.shape = ()
            self.stride = ()
            self.ndim = 0

        else:
            raise RuntimeError(f"Could not infer dtype of type {type(input_data)}")

    def _set_data(self, data: list, shape: tuple, stride: tuple, offset: int) -> None:
        # Validate the types
        expected_types = (
            (data, "data", list), (shape, "shape", tuple), (stride, "stride", tuple), (offset, "offset", int)
        )
        for var, var_name, expected_type in expected_types:
            if not isinstance(var, expected_type):
                raise TypeError(f"{var_name} type must be {type(expected_types)}. Found {type(var)}")

        # Validate the values
        data_size = len(data)
        if not len(shape) == len(stride):
            raise ValueError(f"Length of shape and stride must match. Found shape='{shape}' and stride='{stride}'")
        if not all(val > 0 for val in shape):
            raise ValueError(f"All values in shape must be greater than 0. Found {shape}")

        numel = Buffer._numel(shape)
        if numel > data_size and not any(val == 0 for val in stride):
            raise ValueError(f"Shape {shape} has more elements than data elements: {data_size}")
        if not 0 <= offset < data_size:
            raise ValueError(f"0 <= offset < number of elements. Found offset of {offset} and {data_size} elements")
        if numel + offset > data_size and not any(val == 0 for val in stride):
            raise ValueError(f"Shape {shape} has more elements than data elements: {data_size} - {offset} (offset)")
        if not all(0 <= val <= data_size - offset for val in stride):
            raise ValueError(
                f"0 <= stride[idx] <= len(data) - offset. Found that {stride} is < 0 or > {data_size} - {offset}"
            )

        # Assign the new values
        self.data = data
        self.shape = shape
        self.stride = stride
        self.offset = offset
        self.ndim = len(self.shape)
        self.dtype = Buffer._deduce_dtype(self.data)

    def _index_to_flat_index(self, index: tuple) -> int:
        # From a multidimensional index tuple to its corresponding index in the flat array
        return self.offset + sum([idx * stride for idx, stride in zip(index, self.stride)])

    @staticmethod
    def _numel(shape: tuple):
        # Calculate the total number of elements in a buffer with the specified shape
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        return total_elements

    def numel(self):
        # Calculate the total number of elements in the current buffer based on its shape
        return self._numel(self.shape)

    def __iter__(self) -> Any:
        if self.ndim == 0:
            yield self._get((0,))

        else:
            indices = [0] * self.ndim
            for i in range(self.numel()):
                # Get the value for the current iteration
                yield self._get(tuple(indices))

                # Update indices for the next iteration
                for i in range(self.ndim - 1, -1, -1):
                    if indices[i] < self.shape[i] - 1:
                        indices[i] += 1
                        break
                    else:
                        indices[i] = 0

    def _set(self, index: tuple, value: Any) -> None:
        # This method is intended for internal use by operations that need efficient access to data
        self.data[self._index_to_flat_index(index)] = self.dtype.cast(value)

    def _get(self, index: tuple) -> Union[float, int, bool]:
        # This method is intended for internal use by operations that need efficient access to data
        return self.data[self._index_to_flat_index(index)]

    def is_contiguous(self) -> bool:
        # Check if the data is contiguous in memory
        expected_stride = 1
        for dim, s in zip(reversed(self.shape), reversed(self.stride)):
            if s != expected_stride:
                return False

            expected_stride *= dim

        return True

    def _get_contiguous_data(self) -> list:
        # Create a copy of the list with the right order of the elements
        return [element for element in self]

    @staticmethod
    def _broadcastable(first_buffer: Buffer, second_buffer: Buffer) -> bool:
        for dim_first, dim_second in zip(reversed(first_buffer.shape), reversed(second_buffer.shape)):
            if dim_first != 1 and dim_first != dim_second:
                return False

        return True

    def _validate_input_buffer(self, op: Op, other: Buffer) -> None:
        if not isinstance(op, self.Op):
            raise TypeError(f"op argument is not an Op. Found {type(op)}")

        if not isinstance(other, Buffer):
            raise TypeError(f"One of the inputs is not a Buffer object. Found {type(other)}")

        if self.dtype != other.dtype:
            raise ValueError(f"DType mismatch. Found {self.dtype} and {other.dtype}")

        if self.shape != other.shape:
            raise RuntimeError(f"Shape are not equal. Found {self.shape} and {other.shape}")

    def _execute(self, op: Op, other: Buffer) -> Buffer:
        self._validate_input_buffer(op, other)

        # Create a buffer to store the result
        result = Buffer([])

        # Do the operation
        if op == self.Op.SUM:
            data = [first_element + second_element for first_element, second_element in zip(self, other)]
        elif op == self.Op.SUB:
            data = [first_element - second_element for first_element, second_element in zip(self, other)]
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

        # Assign the data and its metadata to the new buffer
        result._set_data(data=data, shape=self.shape, stride=Buffer._calculate_stride(self.shape), offset=0)

        return result

    def __add__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.SUM, other)

    def __sub__(self, other: Buffer) -> Buffer:
        return self._execute(self.Op.SUB, other)

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
        if not isinstance(new_shape, tuple):
            raise TypeError(f"Expecting type tuple for new_shape but found type {type(new_shape)}")

        if any(value < 1 for value in new_shape):
            raise ValueError(f"One or more values in new_shape is less than 1: {new_shape}")

        # Check if both shapes would have the same number of elements
        if self.numel() != Buffer._numel(new_shape):
            raise RuntimeError(
                "Both current buffer and new buffer has to have the same number of elements." +
                f"Current buffer has {self.numel()} elements and new_shape has {Buffer._numel(new_shape)} elements."
            )

        # Create the output
        result = Buffer([])
        if self.is_contiguous():
            # If the tensor is already contiguous, reshape without creating a copy
            result._set_data(
                data=self.data, shape=new_shape, stride=Buffer._calculate_stride(new_shape), offset=self.offset
            )

        else:
            # Create a contiguous copy of the data before reshaping
            result._set_data(
                data=self._get_contiguous_data(), shape=new_shape, stride=Buffer._calculate_stride(new_shape), offset=0
            )

        return result

    def expand(self, new_shape: tuple) -> Buffer:
        if not isinstance(new_shape, tuple):
            raise TypeError(f"Expecting type tuple for new_shape but found type {type(new_shape)}")

        # Check if the number of dimensions is compatible
        if self.ndim != len(new_shape):
            raise ValueError(f"Number of dimensions must be the same. Found {len(new_shape)} and {self.ndim}")

        # Check if sizes are compatible for expansion
        for idx, (new_size, original_size) in enumerate(zip(new_shape, self.shape)):
            if original_size != 1 and new_size != original_size:
                raise ValueError(f"Sizes are not compatible in dimension {idx}. Found {original_size} and {new_size}")

        # Create the output
        result = Buffer([])
        if self.is_contiguous():
            # The new stride is the old one but we set to 0 where the old shape is 1 and the new shape is not 1
            new_stride = tuple(
                val if self.shape[idx] != 1 or new_shape[idx] == 1 else 0 for idx, val in enumerate(self.stride)
            )
            # If the tensor is already contiguous, expand without creating a copy
            result._set_data(data=self.data, shape=new_shape, stride=new_stride, offset=self.offset)

        else:
            # Because we have to create a contiguous copy of the data, we need the stride of that contiguous data
            contiguous_stride = Buffer._calculate_stride(self.shape)

            # The new stride is contiguous_stride but we set to 0 where the old shape is 1 and the new shape is not 1
            new_stride = tuple(
                val if self.shape[idx] != 1 or new_shape[idx] == 1 else 0 for idx, val in enumerate(contiguous_stride)
            )
            # Create a contiguous copy of the data before expanding
            result._set_data(data=self._get_contiguous_data(), shape=new_shape, stride=new_stride, offset=0)

        return result
