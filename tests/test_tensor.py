import pytest
from tinygpt.tensor import Tensor, DType


def test_DType():

    for value in [-2, -1, 0, 1, 2]:
        assert DType.bool.cast(value) == (value != 0)
        assert DType.float32.cast(value) == float(value)
        assert DType.float32.cast((value != 0)) == float((value != 0))
        assert DType.int32.cast(value) == int(value)
        assert DType.int32.cast((value != 0)) == int((value != 0))


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
        data = [dtype.cast(number) for number in list(range(-3, 3))]

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


def test_tensor_set_data():
    # Create a list of data to set
    data = [0, 1, 2, 3, 4]
    dtype = DType.int32

    valid_arrays = [[float(i) for i in data], [int(i) for i in data], [bool(i) for i in data]]
    valid_shapes = [(1,), (1, 4, 4123, 2), (len(data),)]
    valid_strides = [(1,), (1, 2), (4, 8, 32423)]
    valid_offsets = [0, 1, 15]

    not_valid_shapes = [(-1,), None, (1, -1)]
    not_valid_strides = [(-1,), None, (0,), (1, 0)]
    not_valid_offsets = [-1, None, ()]

    # Try all combinations
    for array_data in valid_arrays:
        tensor = Tensor([])
        for shape in valid_shapes:
            for stride in valid_strides:
                for offset in valid_offsets:
                    tensor._set_data(data=array_data, shape=shape, stride=stride, offset=offset, dtype=dtype)

                    # Check data has been update as expected
                    assert tensor.data == array_data
                    assert tensor.shape == shape
                    assert tensor.stride == stride
                    assert tensor.offset == offset

                    # Try wrong values
                    for not_valid_shape in not_valid_shapes:
                        with pytest.raises(AssertionError):
                            tensor._set_data(
                                data=array_data, shape=not_valid_shape, stride=stride, offset=offset, dtype=dtype
                            )

                    for not_valid_stride in not_valid_strides:
                        with pytest.raises(AssertionError):
                            tensor._set_data(
                                data=array_data, shape=shape, stride=not_valid_stride, offset=offset, dtype=dtype
                            )

                    for not_valid_offset in not_valid_offsets:
                        with pytest.raises(AssertionError):
                            tensor._set_data(
                                data=array_data, shape=shape, stride=stride, offset=not_valid_offset, dtype=dtype
                            )

                    with pytest.raises(AssertionError):
                        tensor._set_data(
                            data=array_data, shape=shape, stride=stride, offset=not_valid_offset, dtype="WrongType"
                        )


def test_tensor_index_to_flat_index():
    data = [
        [[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23]]
    ]
    tensor = Tensor(data)

    flat_idx = 0
    for dim1 in range(4):
        for dim2 in range(2):
            for dim3 in range(3):
                assert flat_idx == tensor._index_to_flat_index((dim1, dim2, dim3))
                flat_idx += 1

    # Try with a different attributes for the underling array
    tensor = Tensor([])
    data = [-1, 0, 1, 2, 3, 4]
    tensor._set_data(data=data, shape=(2, 2), stride=(2, 1), offset=2, dtype=DType.int32)

    assert tensor._index_to_flat_index((0, 0)) == 2
    assert tensor._index_to_flat_index((0, 1)) == 3
    assert tensor._index_to_flat_index((1, 0)) == 4
    assert tensor._index_to_flat_index((1, 1)) == 5


def test_tensor_set():
    # Create the tensor
    tensor = Tensor([])
    data = [-1, -2, -3, -4, -5, -6]
    offset = 2
    shape = (2, 2)
    stride = (2, 1)
    dtype = DType.int32
    tensor._set_data(data=data, shape=shape, stride=stride, offset=offset, dtype=dtype)

    # Modify the data
    new_values = []
    for dim1 in range(shape[0]):
        for dim2 in range(shape[1]):
            new_value = (dim1 + 1) * (dim2 + 1)
            tensor._set((dim1, dim2), new_value)
            new_values.append(new_value)

    for idx, new_value in enumerate(new_values):
        assert tensor.data[offset + idx] == new_value


def test_tensor_get():
    # Create the tensor
    tensor = Tensor([])
    data = [-1, -2, -3, -4, -5, -6]
    offset = 2
    shape = (2, 2)
    stride = (2, 1)
    dtype = DType.int32
    tensor._set_data(data=data, shape=shape, stride=stride, offset=offset, dtype=dtype)

    # Get the data
    idx = 0
    for dim1 in range(shape[0]):
        for dim2 in range(shape[1]):
            assert tensor._get((dim1, dim2)) == data[offset + idx]
            idx += 1
