import pytest
from tinygpt.utils import DType, MultidimensionalIndexGenerator


def test_DType():
    # prefered_order() function
    dtype_order = DType.prefered_order()
    for dtype in DType:
        assert dtype in dtype_order

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

    # type_promotion() function
    for dtype in DType:
        assert DType.type_promotion(dtype, dtype) == dtype

    for dtype in DType:
        assert DType.type_promotion(dtype, DType.float32) == DType.float32

    for dtype in DType:
        assert DType.type_promotion(dtype, DType.float32) in (DType.float32, DType.int32)

    for not_valid_input in [[], (), None]:
        with pytest.raises(RuntimeError, match="first_type <"):
            DType.type_promotion(not_valid_input, not_valid_input)


def test_MultidimensionalIndexGenerator():
    for shape in [(1, 2), (), (1, 1), (1, 1, 2), (2, 1, 1, 2), (2, 3, 1, 2)]:
        expected_num_its = 1
        for value in shape:
            expected_num_its *= max(value, 1)

        num_its = 0
        all_indexes = set()
        for index in MultidimensionalIndexGenerator(shape):
            all_indexes.add(index)

            if len(shape) == 0:
                assert index == (0,)
            else:
                assert all(0 <= value < shape[idx] for idx, value in enumerate(index))

            num_its += 1

        assert num_its == expected_num_its
        assert len(all_indexes) == expected_num_its
