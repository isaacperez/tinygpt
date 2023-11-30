from __future__ import annotations
from typing import Any
from enum import Enum


class DType(Enum):
    # Each type has an equivalent Python type and a priority associated
    float32 = (float, 2)
    int32 = (int, 1)
    bool = (bool, 0)

    def cast(self, input_value: Any) -> Any:
        return self.value[0](input_value)

    @staticmethod
    def deduce_dtype(value: Any) -> DType:
        for dtype in DType:
            if type(value) is dtype.value[0]:
                return dtype
        raise RuntimeError(f"Could not infer dtype of type {type(value)}")

    @staticmethod
    def type_promotion(first_type: DType, second_type: DType):
        if first_type.value[1] > second_type.value[1]:
            return first_type
        else:
            return second_type
