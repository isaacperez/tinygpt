from __future__ import annotations
from typing import Any
from enum import Enum


class DType(Enum):
    float32 = float
    int32 = int
    bool = bool

    def cast(self, input_value: Any) -> Any:
        return self.value(input_value)

    @staticmethod
    def prefered_order():
        return (DType.float32, DType.int32, DType.bool)

    @staticmethod
    def deduce_dtype(value: Any) -> DType:
        for dtype in DType:
            if type(value) is dtype.value:
                return dtype
        raise RuntimeError(f"Could not infer dtype of type {type(value)}")

    @staticmethod
    def type_promotion(first_type: DType, second_type: DType):
        for dtype in DType.prefered_order():
            if first_type == dtype or second_type == dtype:
                return dtype
        raise RuntimeError(f"first_type {type(first_type)} and second_type {type(second_type)} are not DType objects")
