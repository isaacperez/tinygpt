import pytest
from tinygpt.tensor import Tensor, DType


def test_DType():

    for dtype in DType:
        assert isinstance(DType.DType2arrayType(dtype), str)

    with pytest.raises(RuntimeError, match="dtype 'INVENTED_DTYPE' is not a DType object"):
        DType.DType2arrayType("INVENTED_DTYPE")

    for value in [-2, -1, 0, 1, 2]:
        assert DType.cast(value, DType.bool) == (value != 0)
        assert DType.cast(value, DType.float32) == float(value)
        assert DType.cast((value != 0), DType.float32) == float((value != 0))
        assert DType.cast(value, DType.int32) == int(value)
        assert DType.cast((value != 0), DType.int32) == int((value != 0))

    with pytest.raises(RuntimeError, match="dtype 'INVENTED_DTYPE' is not a DType object"):
        DType.cast(None, "INVENTED_DTYPE")


def test_Tensor():

    # Scalars
    tensors_dtype_original_value_original_type = []
    for dtype in DType:
        for i in range(-10, 10):
            tensors_dtype_original_value_original_type.append((Tensor(i, dtype=dtype), dtype, i, int))
            tensors_dtype_original_value_original_type.append((Tensor(float(i), dtype=dtype), dtype, float(i), float))

        tensors_dtype_original_value_original_type.append((Tensor(True, dtype=dtype), dtype, True, bool))
        tensors_dtype_original_value_original_type.append((Tensor(False, dtype=dtype), dtype, False, bool))

    for tensor, dtype, original_value, original_type in tensors_dtype_original_value_original_type:
        assert tensor.shape == ()
        assert tensor.offset == 0
        assert tensor.ndim == 0
        assert tensor.stride == ()
        assert tensor.dtype == dtype
        if dtype is DType.bool:
            assert original_type(tensor.data[0]) == (original_value != 0)
        else:
            assert original_type(tensor.data[0]) == original_value

    # Create tensor from lists
    for dtype in DType:
        data = [DType.cast(number, dtype) for number in list(range(-3, 3))]

        expected_data = [value for value in data]
        expected_shape = [len(data)]
        expected_ndim = 1
        expected_stride = [1]
        for i in range(5):
            tensor = Tensor(data, dtype=dtype)

            assert tensor.dtype == dtype
            assert tensor.offset == 0
            assert tensor.shape == tuple(expected_shape)
            assert tensor.ndim == expected_ndim
            assert tensor.stride == tuple(expected_stride)
            for idx, value in enumerate(tensor.data):
                assert value == expected_data[idx]

            # Update the expected values for next iteration
            expected_data = expected_data * 2
            expected_shape.insert(0, 2)
            expected_stride.insert(0, len(data) * expected_stride[0])
            expected_ndim += 1

            # Add a new the dimension
            data = [data, data]

    for different_length_data in [[[1, 2], [3]], [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11]]]]:
        with pytest.raises(ValueError, match="expected sequence of length"):
            tensor = Tensor(different_length_data)

    for different_type_data in [[[1, 2], 3], [[[1, 2, 3], [4, 5, 6]], [9, 8]]]:
        with pytest.raises(TypeError, match="expected type"):
            tensor = Tensor(different_type_data)

    for different_dtype_data in [None, DType]:
        with pytest.raises(RuntimeError, match="Could not infer dtype of type"):
            tensor = Tensor(different_dtype_data)
