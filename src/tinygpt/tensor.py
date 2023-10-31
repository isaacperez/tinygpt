from __future__ import annotations
from collections import deque
from typing import Any
from array import array
from enum import auto, Enum


class DType(Enum):
    float32 = auto()
    int32 = auto()
    bool = auto()

    @classmethod
    def DType2arrayType(self, dtype: DType) -> str:
        if dtype == DType.float32:
            return 'f'
        elif dtype == DType.int32:
            return 'i'
        elif dtype == DType.bool:
            return 'B'
        elif dtype is DType:
            raise RuntimeError(f"Missing implementation for DType {dtype.value.name}")
        else:
            raise RuntimeError(f"dtype '{dtype}' is not a DType object")

    @classmethod
    def cast(self, data: Any, dtype: DType) -> Any:
        if dtype == DType.float32:
            return float(data)
        elif dtype == DType.int32:
            return int(data)
        elif dtype == DType.bool:
            return bool(data)
        elif dtype is DType:
            raise RuntimeError(f"Missing implementation for DType {dtype.value.name}")
        else:
            raise RuntimeError(f"dtype '{dtype}' is not a DType object")


class Tensor():

    def __init__(
            self,
            data: Any,
            dtype: DType = DType.float32,
            requires_grad=False
    ) -> None:
        # All data ends up in a one-dimensional array but we extract some extra information to handle
        # multidimensional operations
        self._extract_data(data, dtype)
        self.requires_grad = requires_grad

        self.grad = None
        self.grad_fn = None

    def __repr__(self) -> str:
        return f"<Tensor {self.data!r}>"

    def _extract_flat_array_and_shape(self, data: Any, dtype: DType) -> (array, list):
        # The size and type of each element in each dimension must always be the same in the same dimension. We assume 
        # that the first element of a dimension designates the type and size expected for the rest. While we go through 
        # the data object extracting values for the flat array, we make the corresponding checks.
        flat_array = array(DType.DType2arrayType(dtype))
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
                flat_array.append(DType.cast(current_element, dtype))

        return flat_array, tuple([value for value in size_by_dim.values() if value != -1])

    def _calculate_stride(self):
        # Creates a tuple with the number of elements of the flat array to be skipped in each dimension to traverse it
        stride = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            stride[i] = stride[i + 1] * self.shape[i + 1]

        return tuple(stride)

    def _extract_data(self, data: Any, dtype: DType) -> None:
        self.dtype = dtype
        self.offset = 0

        if isinstance(data, (list, tuple)):
            self.data, self.shape = self._extract_flat_array_and_shape(data, dtype)
            self.ndim = len(self.shape)
            self.stride = self._calculate_stride()

        elif isinstance(data, (float, int, bool)):
            self.data = array(DType.DType2arrayType(dtype), [DType.cast(data, dtype)])
            self.shape = ()
            self.ndim = 0
            self.stride = ()

        else:
            raise RuntimeError(f"Could not infer dtype of type {type(data)}")
