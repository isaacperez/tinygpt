import pytest
from tinygpt.buffer import Buffer
from tinygpt.utils import DType


def test_Tensor():
    # Scalars
    buffers_dtype_original_value_original_type = []
    for dtype in DType:
        for i in range(-10, 10):
            buffers_dtype_original_value_original_type.append((Buffer(i, dtype=dtype), dtype, i, int))
            buffers_dtype_original_value_original_type.append((Buffer(float(i), dtype=dtype), dtype, float(i), float))

        buffers_dtype_original_value_original_type.append((Buffer(True, dtype=dtype), dtype, True, bool))
        buffers_dtype_original_value_original_type.append((Buffer(False, dtype=dtype), dtype, False, bool))

    for buffer, dtype, original_value, original_type in buffers_dtype_original_value_original_type:
        assert buffer.shape == ()
        assert buffer.offset == 0
        assert buffer.ndim == 0
        assert buffer.stride == ()
        assert buffer.dtype == dtype
        if dtype is DType.bool:
            assert original_type(buffer.data[0]) == (original_value != 0)
        else:
            assert original_type(buffer.data[0]) == original_value

    # Create buffer from lists
    for dtype in DType:
        data = [dtype.cast(number) for number in list(range(-3, 3))]

        expected_data = [value for value in data]
        expected_shape = [len(data)]
        expected_ndim = 1
        expected_stride = [1]
        for i in range(5):
            buffer = Buffer(data, dtype=dtype)

            assert buffer.dtype == dtype
            assert buffer.offset == 0
            assert buffer.shape == tuple(expected_shape)
            assert buffer.ndim == expected_ndim
            assert buffer.stride == tuple(expected_stride)
            for idx, value in enumerate(buffer.data):
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
            buffer = Buffer(different_length_data)

    for different_type_data in [[[1, 2], 3], [[[1, 2, 3], [4, 5, 6]], [9, 8]]]:
        with pytest.raises(TypeError, match="expected type"):
            buffer = Buffer(different_type_data)

    for different_dtype_data in [None, DType]:
        with pytest.raises(RuntimeError, match="Could not infer dtype of type"):
            buffer = Buffer(different_dtype_data)


def test_buffer_set_data():
    # Create a list of data to set
    data = [i for i in range(40)]

    valid_arrays = [[float(i) for i in data], [int(i) for i in data], [bool(i) for i in data]]
    valid_shapes = [(1,), (1, 4, 2, 2), (len(data),)]
    valid_strides = [(1,), (8, 4, 2, 1), (1,)]
    valid_offsets = [4, 8, 0]

    not_valid_shapes = [(-1,), (1, -1), (1, 4, 4123, 2)]
    not_valid_strides = [(-1,), (1, 0, -1), (-1, 0), (1, 2, -1), (1, 4, 8, 32423)]
    not_valid_offsets = [-1, -2]

    # Try wrong data types
    buffer = Buffer([])
    for wrong_data_type in [None, (), "a", -1, 0, 1]:
        with pytest.raises(TypeError):
            buffer._set_data(data=wrong_data_type, shape=(0,), stride=(1,), offset=0)

    for wrong_data_type in [None, [], "a", -1, 0, 1]:
        with pytest.raises(TypeError):
            buffer._set_data(data=data, shape=wrong_data_type, stride=(1,), offset=0)

    for wrong_data_type in [None, [], "a", -1, 0, 1]:
        with pytest.raises(TypeError):
            buffer._set_data(data=data, shape=(0,), stride=wrong_data_type, offset=0)

    for wrong_data_type in [None, (), "a"]:
        with pytest.raises(TypeError):
            buffer._set_data(data=data, shape=(0,), stride=(1,), offset=wrong_data_type)

    # Try different combinations
    for array_data in valid_arrays:
        buffer = Buffer([])
        for shape, stride, offset in zip(valid_shapes, valid_strides, valid_offsets):
            # Assign the values
            buffer._set_data(data=array_data, shape=shape, stride=stride, offset=offset)

            # Check data has been update as expected
            assert buffer.data == array_data
            assert buffer.shape == shape
            assert buffer.stride == stride
            assert buffer.offset == offset

            # Try wrong values
            for not_valid_shape in not_valid_shapes:
                with pytest.raises(ValueError):
                    buffer._set_data(data=array_data, shape=not_valid_shape, stride=stride, offset=offset)

            for not_valid_stride in not_valid_strides:
                with pytest.raises(ValueError):
                    buffer._set_data(data=array_data, shape=shape, stride=not_valid_stride, offset=offset)

            for not_valid_offset in not_valid_offsets:
                with pytest.raises(ValueError):
                    buffer._set_data(data=array_data, shape=shape, stride=stride, offset=not_valid_offset)


def test_buffer_index_to_flat_index():
    data = [
        [[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23]]
    ]
    buffer = Buffer(data)

    flat_idx = 0
    for dim1 in range(4):
        for dim2 in range(2):
            for dim3 in range(3):
                assert flat_idx == buffer._index_to_flat_index((dim1, dim2, dim3))
                flat_idx += 1

    # Try with a different attributes for the underling array
    buffer = Buffer([])
    data = [-1, 0, 1, 2, 3, 4]
    buffer._set_data(data=data, shape=(2, 2), stride=(2, 1), offset=2)

    assert buffer._index_to_flat_index((0, 0)) == 2
    assert buffer._index_to_flat_index((0, 1)) == 3
    assert buffer._index_to_flat_index((1, 0)) == 4
    assert buffer._index_to_flat_index((1, 1)) == 5


def test_buffer_set():
    # Create the buffer
    buffer = Buffer([])
    data = [-1, -2, -3, -4, -5, -6]
    offset = 2
    shape = (2, 2)
    stride = (2, 1)
    buffer._set_data(data=data, shape=shape, stride=stride, offset=offset)

    # Modify the data
    new_values = []
    for dim1 in range(shape[0]):
        for dim2 in range(shape[1]):
            new_value = (dim1 + 1) * (dim2 + 1)
            buffer._set((dim1, dim2), new_value)
            new_values.append(new_value)

    for idx, new_value in enumerate(new_values):
        assert buffer.data[offset + idx] == new_value

    # Use a scalar buffer
    scalar = Buffer(1)
    scalar._set((0,), 2)
    assert scalar.data[0] == 2


def test_buffer_get():
    # Create the buffer
    buffer = Buffer([])
    data = [-1, -2, -3, -4, -5, -6]
    offset = 2
    shape = (2, 2)
    stride = (2, 1)
    buffer._set_data(data=data, shape=shape, stride=stride, offset=offset)

    # Get the data
    idx = 0
    for dim1 in range(shape[0]):
        for dim2 in range(shape[1]):
            assert buffer._get((dim1, dim2)) == data[offset + idx]
            idx += 1

    # Use a scalar buffer
    scalar = Buffer(2)
    assert scalar._get((0,)) == 2


def test_broadcastable():
    # Same shapes
    assert Buffer._broadcastable(Buffer([]), Buffer([]))
    assert Buffer._broadcastable(Buffer(False), Buffer(2))
    assert Buffer._broadcastable(Buffer(True), Buffer(1))
    assert Buffer._broadcastable(Buffer([1, 2, 3]), Buffer([4, 5, 6]))
    assert Buffer._broadcastable(Buffer([[1, 2, 3], [4, 5, 6]]), Buffer([[-1, -2, -3], [-4, -5, -6]]))

    # Broadcasteable
    data = [1]
    for i in range(10):
        assert Buffer._broadcastable(Buffer(1), Buffer(data))
        data.append(i)

    base_data = [[1], [2], [3], [4]]
    base_buffer = Buffer(base_data)
    current_data = base_data
    for _ in range(10):
        current_data = [base_data]
        current_buffer = Buffer(current_data)
        assert Buffer._broadcastable(base_buffer, current_buffer)
        assert Buffer._broadcastable(current_buffer, base_buffer)

    assert Buffer._broadcastable(base_buffer, Buffer([1, 2, 3, 4]))
    assert Buffer._broadcastable(Buffer([[[1], [2]], [[3], [4]]]), Buffer([[[1], [2]], [[3], [4]]]))

    # Different shapes
    assert not Buffer._broadcastable(Buffer([]), Buffer([1, 2]))
    assert not Buffer._broadcastable(Buffer([1, 2]), Buffer([1, 2, 3]))
    assert not Buffer._broadcastable(Buffer([1, 2, 3, 4]), Buffer([1, 2, 3]))
    assert not Buffer._broadcastable(Buffer([[1, 2, 3, 4]]), Buffer([[1, 2, 3]]))
    assert not Buffer._broadcastable(Buffer([[[1], [2]], [[3], [4]], [[3], [4]]]), Buffer([[[1], [2]], [[3], [4]]]))


def test_is_contiguous():
    # Contiguous array
    assert Buffer(True).is_contiguous()
    assert Buffer(1).is_contiguous()

    data = [1, 2, 3, 4, 5, 6]
    buffer = Buffer(data)
    assert buffer.is_contiguous()

    shape = (1, 4, 1)
    stride = (0, 1, 0)
    for i in range(2):
        buffer._set_data(data=data, shape=shape, stride=stride, offset=i)
        assert buffer.is_contiguous()

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    shape = (1, 2, 2, 1)
    stride = (0, 2, 1, 0)
    for i in range(3):
        buffer._set_data(data=data, shape=shape, stride=stride, offset=i)
        assert buffer.is_contiguous()

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    # Non-contiguous array
    data = [1, 2, 3, 4]
    buffer = Buffer(data)

    buffer._set_data(data=data, shape=(2,), stride=(2,), offset=0)
    assert not buffer.is_contiguous()

    buffer._set_data(data=data, shape=(1,), stride=(4,), offset=0)
    assert not buffer.is_contiguous()

    buffer._set_data(data=data, shape=(1, 1), stride=(2, 2), offset=0)
    assert not buffer.is_contiguous()

    # Transpose an array
    data = [i for i in range(12)]
    buffer = Buffer(data)
    buffer._set_data(data=data, shape=(3, 4), stride=(4, 1), offset=0)
    assert buffer.is_contiguous()
    buffer._set_data(data=data, shape=(4, 3), stride=(1, 4), offset=0)
    assert not buffer.is_contiguous()


def test_get_contiguous_data():
    # Contiguous array
    assert Buffer(True)._get_contiguous_data() == [True]
    assert Buffer(1)._get_contiguous_data() == [1]

    data = [1, 2, 3, 4, 5, 6]
    buffer = Buffer(data)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == data
    assert id(contiguous_data) != id(data)

    shape = (1, 4, 1)
    stride = (0, 1, 0)
    for i in range(2):
        buffer._set_data(data=data, shape=shape, stride=stride, offset=i)
        contiguous_data = buffer._get_contiguous_data()
        assert contiguous_data == data[i:i+4]
        assert id(contiguous_data) != id(data)

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    shape = (1, 2, 2, 1)
    stride = (0, 2, 1, 0)
    for i in range(3):
        buffer._set_data(data=data, shape=shape, stride=stride, offset=i)
        contiguous_data = buffer._get_contiguous_data()
        assert contiguous_data == data[i:i+4]
        assert id(contiguous_data) != id(data)

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    # Non-contiguous array
    data = [1, 2, 3, 4]
    buffer = Buffer(data)

    buffer._set_data(data=data, shape=(2,), stride=(2,), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [1, 3]
    assert id(contiguous_data) != id(data)

    buffer._set_data(data=data, shape=(1,), stride=(4,), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [1]
    assert id(contiguous_data) != id(data)

    buffer._set_data(data=data, shape=(1, 1), stride=(2, 2), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [1]
    assert id(contiguous_data) != id(data)

    # Transpose an array
    data = [i for i in range(12)]
    buffer = Buffer(data)
    buffer._set_data(data=data, shape=(3, 4), stride=(4, 1), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == data
    assert id(contiguous_data) != id(data)

    buffer._set_data(data=data, shape=(4, 3), stride=(1, 4), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [j * 4 + i for i in range(4) for j in range(3)]
    assert id(contiguous_data) != id(data)


def test_numel():
    data = [i for i in range(24)]
    buffer = Buffer(data)
    assert buffer.numel() == len(data)

    for i in range(3):
        buffer._set_data(data=data, shape=(1, 1, 1), stride=(i, i, i), offset=i)
        assert buffer.numel() == 1

        buffer._set_data(data=data, shape=(2,), stride=(i,), offset=i)
        assert buffer.numel() == 2

        buffer._set_data(data=data, shape=(2, 2), stride=(i, 1), offset=i)
        assert buffer.numel() == 4

        buffer._set_data(data=data, shape=(3, 2), stride=(2, 1), offset=i)
        assert buffer.numel() == 6

    assert Buffer(True).numel() == 1
    assert Buffer(0).numel() == 1
    assert Buffer(-3.14).numel() == 1


def test_it():
    numel = 24
    data = [i for i in range(numel)]
    buffer = Buffer(data)
    idx = 0
    for element in buffer:
        assert element == data[idx]
        idx += 1
    assert idx == numel

    # Change the offset
    for offset in range(10):
        buffer._set_data(data=data, shape=(numel - offset,), stride=(1,), offset=offset)
        idx = 0
        for element in buffer:
            assert element == data[offset + idx]
            idx += 1
        assert idx == numel - offset

    # Add empty dimensions
    shape = (numel,)
    stride = (1,)
    for _ in range(5):
        shape = (1, *shape, 1)
        stride = (1, *stride, 1)
        buffer._set_data(data=data, shape=shape, stride=stride, offset=0)
        idx = 0
        for element in buffer:
            assert element == data[idx]
            idx += 1
        assert idx == numel

    # Multidimensional buffer
    numel = 24
    data = [i for i in range(numel)]
    buffer = Buffer(data)
    buffer._set_data(data=data, shape=(2, 2), stride=(2, 1), offset=0)
    idx = 0
    for element in buffer:
        assert element == data[idx]
        idx += 1
    assert idx == 4

    for offset in range(2):
        buffer._set_data(data=data, shape=(2, 2), stride=(1, 2), offset=offset)
        idx = 0
        expected_output = [0, 2, 1, 3]
        for element in buffer:
            assert element == expected_output[idx] + offset
            idx += 1
        assert idx == 4
