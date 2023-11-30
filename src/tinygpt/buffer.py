from __future__ import annotations
from collections import deque
from typing import Any, Union

from tinygpt.utils import DType


class Buffer():

    def __init__(
            self,
            data: Any,
            dtype: DType = None
    ) -> None:
        # All data ends up in a one-dimensional array but we extract some metadata to handle multidimensional operations
        self.data, self.offset, self.ndim, self.stride, self.shape, self.dtype = self._extract_data(data, dtype)

    def __repr__(self) -> str:
        if self.ndim > 1 and any(value == 0 for value in self.shape):
            return f"<Buffer {self._get_contiguous_data()}, size={self.shape}>"
        else:
            return f"<Buffer {self._get_contiguous_data()}>"

    def __str__(self) -> str:
        return f"{self._get_contiguous_data()}"

    def _extract_flat_array_and_shape(self, data: Any, dtype: DType) -> (list, list):
        # The size and type of each element in each dimension must always be the same in the same dimension. We assume
        # that the first element of a dimension designates the type and size expected for the rest. While we go through
        # the data object extracting values for the flat array, we make the corresponding checks.
        flat_array = []
        size_by_dim = {}
        type_by_dim = {}
        queue = deque([(data, 1)])
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

    def _extract_data(self, data: Any, dtype: DType | None) -> None:
        offset = 0

        if dtype is None:
            dtype = self._deduce_dtype(data)

        if isinstance(data, (list, tuple)):
            data, shape = self._extract_flat_array_and_shape(data, dtype)
            stride = self._calculate_stride(shape)
            ndim = len(shape)

        elif isinstance(data, (float, int, bool)):
            data = [dtype.cast(data)]
            shape = ()
            stride = ()
            ndim = 0

        else:
            raise RuntimeError(f"Could not infer dtype of type {type(data)}")

        return data, offset, ndim, stride, shape, dtype

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
        if numel > data_size:
            raise ValueError(f"Shape {shape} has more elements than data elements: {data_size}")
        if not 0 <= offset < data_size:
            raise ValueError(f"0 <= offset < number of elements. Found offset of {offset} and {data_size} elements")
        if numel + offset > data_size:
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
        # This method is intended for internal use by operations that need efficient access to data. It should not
        # be used when manipulating a buffer. Instead, when you want to access/modify the elements of the buffer do it
        # as it is usually done in Python using the __getitem__ method (via the `[]` operator).
        self.data[self._index_to_flat_index(index)] = self.dtype.cast(value)

    def _get(self, index: tuple) -> Union[float, int, bool]:
        # This method is intended for internal use by operations that need efficient access to data. It should not
        # be used when manipulating a buffer. Instead, when you want to access/modify the elements of the buffer do it
        # as it is usually done in Python using the __getitem__ method (via the `[]` operator).
        return self.data[self._index_to_flat_index(index)]

    def is_contiguous(self) -> bool:
        # Check if the data is contiguous in memory
        expected_stride = 1
        for dim, s in zip(reversed(self.shape), reversed(self.stride)):
            if s != expected_stride and s != 0:
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
