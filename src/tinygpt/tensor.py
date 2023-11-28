from __future__ import annotations
from collections import deque
from typing import Any, Union

from tinygpt.utils import DType, MultidimensionalIndexGenerator


class Tensor():

    def __init__(
            self,
            data: Any,
            dtype: DType = None,
            requires_grad=False
    ) -> None:
        # All data ends up in a one-dimensional array but we extract some metadata to handle multidimensional operations
        self.data, self.offset, self.ndim, self.stride, self.shape, self.dtype = self._extract_data(data, dtype)

        # Gradient-related metadata
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def __repr__(self) -> str:
        if self.ndim > 1 and any(value == 0 for value in self.shape):
            return f"<Tensor {self.data!r}, size={self.shape}>"
        else:
            return f"<Tensor {self.data!r}>"

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
    def _calculate_stride(shape):
        # Creates a tuple with the number of elements of the flat array to be skipped in each dimension to traverse it
        ndim = len(shape)
        stride = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]

        return tuple(stride)

    @staticmethod
    def _deduce_dtype(data):
        # If it's a list find the first element if there is one
        first_element = data
        while isinstance(first_element, (tuple, list)) and len(first_element) > 0:
            first_element = first_element[0]

        # By default, an empty tensor use DType.float32. If there is a value we deduce its DType
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
        # Assigns the flat array
        assert isinstance(data, list)
        assert isinstance(shape, tuple)
        assert all(val > 0 for val in shape)
        assert isinstance(stride, tuple)
        assert all(val >= 0 for val in stride)
        assert len(shape) == len(stride)
        assert isinstance(offset, int)
        assert offset >= 0

        self.data = data
        self.shape = shape
        self.stride = stride
        self.offset = offset
        self.ndim = len(self.shape)
        self.dtype = Tensor._deduce_dtype(self.data)

    def _index_to_flat_index(self, index: tuple) -> int:
        return self.offset + sum([idx * stride for idx, stride in zip(index, self.stride)])

    def _set(self, index: tuple, value: Any) -> None:
        # This method is intended for internal use by operations that need efficient access to data. It should not
        # be used when manipulating a tensor. Instead, when you want to access/modify the elements of the tensor do it
        # as it is usually done in Python using the __getitem__ method (via the `[]` operator).
        self.data[self._index_to_flat_index(index)] = self.dtype.cast(value)

    def _get(self, index: tuple) -> Union[float, int, bool]:
        # This method is intended for internal use by operations that need efficient access to data. It should not
        # be used when manipulating a tensor. Instead, when you want to access/modify the elements of the tensor do it
        # as it is usually done in Python using the __getitem__ method (via the `[]` operator).
        return self.data[self._index_to_flat_index(index)]

    def is_contiguous(self) -> bool:
        # Check if the tensor's data is contiguous in memory
        expected_stride = 1
        for dim, s in zip(reversed(self.shape), reversed(self.stride)):
            if s != expected_stride and s != 0:
                return False
            expected_stride *= dim
        return True

    def _get_contiguous_data(self):
        if self.is_contiguous():
            # If the tensor is already contiguous, return the internal data
            return self.data
        else:
            # Create a copy of the list
            return [self._get(idx) for idx in MultidimensionalIndexGenerator(self.shape)]

    @staticmethod
    def _broadcastable(first_tensor, second_tensor):
        for dim_first, dim_second in zip(reversed(first_tensor.shape), reversed(second_tensor.shape)):
            if dim_first != 1 and dim_first != dim_second:
                return False
        return True
