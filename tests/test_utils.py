import pytest

from tinygpt.utils import DType


def test_DType():

    # cast() method
    for value in [-2, -1, 0, 1, 2]:
        assert DType.bool.cast(value) == (value != 0)
        assert DType.float32.cast(value) == float(value)
        assert DType.float32.cast((value != 0)) == float((value != 0))
        assert DType.int32.cast(value) == int(value)
        assert DType.int32.cast((value != 0)) == int((value != 0))

    # deduce_dtype() function
    for int_val in range(-5, 5):
        assert DType.deduce_dtype(float(int_val)) is DType.float32

    for int_val in range(-5, 5):
        assert DType.deduce_dtype(int_val) is DType.int32

    for bool_val in [True, False]:
        assert DType.deduce_dtype(bool_val) is DType.bool

    for not_valid_input in [[], (), None]:
        with pytest.raises(RuntimeError, match="Could not infer dtype of type"):
            DType.deduce_dtype(not_valid_input)
