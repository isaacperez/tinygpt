from __future__ import annotations
from collections import deque
from typing import Union, Any
from array import array
from enum import auto, Enum


class DType(Enum):
    float32 = auto()
    int64 = auto()
    bool = auto()

    @classmethod
    def DType2arrayType(self, dtype: DType) -> str:
        if dtype == DType.float32:
            return 'f'
        elif dtype == DType.int64:
            return 'I'
        elif dtype == DType.bool:
            return 'B'
        else:
            raise RuntimeError(f"Missing implementation for DType {dtype.value.name}")

    @classmethod
    def cast(self, data: Any, dtype: DType) -> Any:
        if dtype == DType.float32:
            return float(data)
        elif dtype == DType.int64:
            return int(data)
        elif dtype == DType.bool:
            return bool(data)
        else:
            raise RuntimeError(f"Missing implementation for DType {dtype.value.name}")


class Tensor():

    def __init__(
            self,
            data: Union[list, tuple, float, int, bool],
            dtype: DType = DType.float32,
            requires_grad=False
    ) -> None:
        # All data ends up in a one-dimensional array but we extract some extra information to handle
        # multidimensional indexing
        self.set_data(data, dtype)
        self.requires_grad = requires_grad

        self.grad = None
        self.grad_fn = None

    def __repr__(self) -> str:
        return f"<Tensor {self.data!r}>"

    def _extract_flat_array_and_shape(self, data, dtype) -> (array, list):
        flat_array = []
        size_by_dim = {}
        type_by_dim = {}
        queue = deque([(data, 1)])
        while len(queue) > 0:
            current_element, dim = queue.popleft()
            current_element_is_a_list = isinstance(current_element, (list, tuple))

            if current_element_is_a_list:
                size_by_dim[dim] = size_by_dim.get(dim, len(current_element))

            type_by_dim[dim] = type_by_dim.get(dim, type(current_element))

            if current_element_is_a_list and len(current_element) != size_by_dim[dim]:
                raise ValueError(
                    f"expected sequence of length {size_by_dim[dim]} at dim {dim} (got {len(current_element)})"
                )

            elif not isinstance(current_element, type_by_dim[dim]):
                raise TypeError(f"expected type {type_by_dim[dim]} at dim {dim}, not {type(current_element)}")

            if current_element_is_a_list:
                for element in current_element:
                    queue.append((element, dim + 1))
            else:
                flat_array.append(DType.cast(current_element, dtype))

        return flat_array, tuple(size_by_dim.values())

    def _calculate_stride(self):
        strides = [1] * self.ndim

        for i in range(self.ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]

        return tuple(strides)

    def set_data(self, data, dtype):
        self.dtype = dtype
        self.offset = 0

        if isinstance(data, (list, tuple)):
            self.data, self.shape = self._extract_flat_array_and_shape(data, dtype)
            self.ndim = len(self.shape)
            self.stride = self._calculate_stride()

        elif isinstance(data, (float, int, bool)):
            self.ndim = 0
            self.stride = ()
            self.shape = ()
            self.data = array(DType.DType2arrayType(dtype), [DType.cast(data, dtype)])

        else:
            raise RuntimeError(f"Could not infer dtype of type {type(data)}")
