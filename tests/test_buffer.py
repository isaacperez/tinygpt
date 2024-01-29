import math

import pytest

from tinygpt.buffer import Buffer
from tinygpt.utils import DType


def test_Buffer():
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
        assert buffer.numel == 1
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
        expected_numel = len(data)
        for i in range(5):
            buffer = Buffer(data, dtype=dtype)

            assert buffer.dtype == dtype
            assert buffer.offset == 0
            assert buffer.shape == tuple(expected_shape)
            assert buffer.ndim == expected_ndim
            assert buffer.stride == tuple(expected_stride)
            assert buffer.numel == expected_numel
            for idx, value in enumerate(buffer.data):
                assert value == expected_data[idx]

            # Update the expected values for next iteration
            expected_data = expected_data * 2
            expected_shape.insert(0, 2)
            expected_stride.insert(0, len(data) * expected_stride[0])
            expected_ndim += 1
            expected_numel *= 2

            # Add a new the dimension
            data = [data, data]

    for different_length_data in [[[1, 2], [3]], [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11]]]]:
        with pytest.raises(ValueError, match="Inconsistent size at dim"):
            buffer = Buffer(different_length_data)

    for different_type_data in [[[1, 2], 3], [[[1, 2, 3], [4, 5, 6]], [9, 8]]]:
        with pytest.raises(TypeError, match="Type mismatch at dim"):
            buffer = Buffer(different_type_data)

    for different_dtype_data in [None, DType]:
        with pytest.raises(RuntimeError, match="Could not infer dtype of type"):
            buffer = Buffer(different_dtype_data)

    # Test copy constructor
    for data in [1, -1.0, True, False, 0, 0.0, [1, 2], [], [[], []], [[[1], [2], [3]], [[4], [5], [6]]]]:
        original_buffer = Buffer(data)
        copy_buffer = Buffer(original_buffer)

        assert original_buffer.data == copy_buffer.data
        assert id(original_buffer.data) == id(copy_buffer.data)
        assert original_buffer.offset == copy_buffer.offset
        assert original_buffer.ndim == copy_buffer.ndim
        assert original_buffer.stride == copy_buffer.stride
        assert original_buffer.shape == copy_buffer.shape
        assert original_buffer.dtype == copy_buffer.dtype
        assert original_buffer.numel == copy_buffer.numel

        # Try to do a casting when copying the buffer
        for dtype in DType:
            if dtype != original_buffer.dtype:
                with pytest.raises(RuntimeError, match="dtype doesn't match, and casting isn't supported"):
                    copy_buffer = Buffer(original_buffer, dtype=dtype)
            else:
                copy_buffer = Buffer(original_buffer, dtype=dtype)

                assert original_buffer.data == copy_buffer.data
                assert id(original_buffer.data) == id(copy_buffer.data)
                assert original_buffer.offset == copy_buffer.offset
                assert original_buffer.ndim == copy_buffer.ndim
                assert original_buffer.stride == copy_buffer.stride
                assert original_buffer.shape == copy_buffer.shape
                assert original_buffer.dtype == copy_buffer.dtype
                assert original_buffer.numel == copy_buffer.numel


def test_buffer_create_buffer_from_data():
    # Create a list of data to set
    data = [i for i in range(40)]

    valid_arrays = [[float(i) for i in data], [int(i) for i in data], [bool(i) for i in data]]
    valid_shapes = [(1,), (1, 4, 2, 2), (len(data),)]
    valid_strides = [(1,), (8, 4, 2, 1), (1,)]
    valid_offsets = [4, 8, 0]
    numels = [1, 16, 40]

    not_valid_shapes = [(-1,), (1, -1), (1, 4, 4123, 2)]
    not_valid_strides = [(-1,), (1, 0, -1), (-1, 0), (1, 2, -1), (1, 4, 8, 32423)]
    not_valid_offsets = [-1, -2]

    # Try wrong data types
    buffer = Buffer([])
    for wrong_data_type in [None, (), "a", -1, 0, 1]:
        with pytest.raises(TypeError):
            Buffer._create_buffer_from_data(data=wrong_data_type, shape=(0,), stride=(1,), offset=0)

    for wrong_data_type in [None, [], "a", -1, 0, 1]:
        with pytest.raises(TypeError):
            Buffer._create_buffer_from_data(data=data, shape=wrong_data_type, stride=(1,), offset=0)

    for wrong_data_type in [None, [], "a", -1, 0, 1]:
        with pytest.raises(TypeError):
            Buffer._create_buffer_from_data(data=data, shape=(0,), stride=wrong_data_type, offset=0)

    for wrong_data_type in [None, (), "a"]:
        with pytest.raises(TypeError):
            Buffer._create_buffer_from_data(data=data, shape=(0,), stride=(1,), offset=wrong_data_type)

    # Try different combinations
    for array_data in valid_arrays:
        for shape, stride, offset, numel in zip(valid_shapes, valid_strides, valid_offsets, numels):
            # Assign the values
            buffer = Buffer._create_buffer_from_data(data=array_data, shape=shape, stride=stride, offset=offset)

            # Check data has been update as expected
            assert buffer.data == array_data
            assert buffer.shape == shape
            assert buffer.stride == stride
            assert buffer.offset == offset
            assert buffer.numel == numel

            # Try wrong values
            for not_valid_shape in not_valid_shapes:
                with pytest.raises(ValueError):
                    Buffer._create_buffer_from_data(
                        data=array_data, shape=not_valid_shape, stride=stride, offset=offset
                    )

            for not_valid_stride in not_valid_strides:
                with pytest.raises(ValueError):
                    Buffer._create_buffer_from_data(
                        data=array_data, shape=shape, stride=not_valid_stride, offset=offset
                    )

            for not_valid_offset in not_valid_offsets:
                with pytest.raises(ValueError):
                    Buffer._create_buffer_from_data(
                        data=array_data, shape=shape, stride=stride, offset=not_valid_offset
                    )


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
    data = [-1, 0, 1, 2, 3, 4]
    buffer = Buffer._create_buffer_from_data(data=data, shape=(2, 2), stride=(2, 1), offset=2)

    assert buffer._index_to_flat_index((0, 0)) == 2
    assert buffer._index_to_flat_index((0, 1)) == 3
    assert buffer._index_to_flat_index((1, 0)) == 4
    assert buffer._index_to_flat_index((1, 1)) == 5


def test_buffer_set():
    # Create the buffer
    data = [-1, -2, -3, -4, -5, -6]
    offset = 2
    shape = (2, 2)
    stride = (2, 1)
    buffer = Buffer._create_buffer_from_data(data=data, shape=shape, stride=stride, offset=offset)

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
    data = [-1, -2, -3, -4, -5, -6]
    offset = 2
    shape = (2, 2)
    stride = (2, 1)
    buffer = Buffer._create_buffer_from_data(data=data, shape=shape, stride=stride, offset=offset)

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

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    buffer = Buffer(data)
    assert buffer.is_contiguous()

    shape = (1, 4, 3)
    for i in range(2):
        buffer = Buffer._create_buffer_from_data(
            data=data, shape=shape, stride=Buffer._calculate_stride(shape), offset=i
        )

        assert buffer.is_contiguous()
        shape = (1, *shape, 1)

    shape = (1, 2, 3, 1)
    for i in range(3):
        buffer = Buffer._create_buffer_from_data(
            data=data, shape=shape, stride=Buffer._calculate_stride(shape), offset=i
        )

        assert buffer.is_contiguous()
        shape = (1, *shape, 1)

    # Non-contiguous array
    buffer = Buffer(data)
    assert buffer.is_contiguous()

    shape = (1, 4, 1)
    stride = (0, 1, 0)
    for i in range(2):
        buffer = Buffer._create_buffer_from_data(data=data, shape=shape, stride=stride, offset=i)
        assert not buffer.is_contiguous()

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    shape = (1, 2, 2, 1)
    stride = (0, 2, 1, 0)
    for i in range(3):
        buffer = Buffer._create_buffer_from_data(data=data, shape=shape, stride=stride, offset=i)
        assert not buffer.is_contiguous()

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    data = [1, 2, 3, 4]

    buffer = Buffer._create_buffer_from_data(data=data, shape=(2,), stride=(2,), offset=0)
    assert not buffer.is_contiguous()

    buffer = Buffer._create_buffer_from_data(data=data, shape=(1,), stride=(4,), offset=0)
    assert not buffer.is_contiguous()

    buffer = Buffer._create_buffer_from_data(data=data, shape=(1, 1), stride=(2, 2), offset=0)
    assert not buffer.is_contiguous()

    # Transpose an array
    data = [i for i in range(12)]
    buffer = Buffer._create_buffer_from_data(data=data, shape=(3, 4), stride=(4, 1), offset=0)
    assert buffer.is_contiguous()
    buffer = Buffer._create_buffer_from_data(data=data, shape=(4, 3), stride=(1, 4), offset=0)
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
        buffer = Buffer._create_buffer_from_data(data=data, shape=shape, stride=stride, offset=i)
        contiguous_data = buffer._get_contiguous_data()
        assert contiguous_data == data[i:i+4]
        assert id(contiguous_data) != id(data)

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    shape = (1, 2, 2, 1)
    stride = (0, 2, 1, 0)
    for i in range(3):
        buffer = Buffer._create_buffer_from_data(data=data, shape=shape, stride=stride, offset=i)
        contiguous_data = buffer._get_contiguous_data()
        assert contiguous_data == data[i:i+4]
        assert id(contiguous_data) != id(data)

        shape = (1, *shape, 1)
        stride = (0, *stride, 0)

    # Non-contiguous array
    data = [1, 2, 3, 4]
    buffer = Buffer(data)

    buffer = Buffer._create_buffer_from_data(data=data, shape=(2,), stride=(2,), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [1, 3]
    assert id(contiguous_data) != id(data)

    buffer = Buffer._create_buffer_from_data(data=data, shape=(1,), stride=(4,), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [1]
    assert id(contiguous_data) != id(data)

    buffer = Buffer._create_buffer_from_data(data=data, shape=(1, 1), stride=(2, 2), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [1]
    assert id(contiguous_data) != id(data)

    # Transpose an array
    data = [i for i in range(12)]
    buffer = Buffer(data)
    buffer = Buffer._create_buffer_from_data(data=data, shape=(3, 4), stride=(4, 1), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == data
    assert id(contiguous_data) != id(data)

    buffer = Buffer._create_buffer_from_data(data=data, shape=(4, 3), stride=(1, 4), offset=0)
    contiguous_data = buffer._get_contiguous_data()
    assert contiguous_data == [j * 4 + i for i in range(4) for j in range(3)]
    assert id(contiguous_data) != id(data)


def test_numel():
    data = [i for i in range(24)]
    buffer = Buffer(data)
    assert buffer.numel == len(data)

    for i in range(3):
        buffer = Buffer._create_buffer_from_data(data=data, shape=(1, 1, 1), stride=(i, i, i), offset=i)
        assert buffer.numel == 1

        buffer = Buffer._create_buffer_from_data(data=data, shape=(2,), stride=(i,), offset=i)
        assert buffer.numel == 2

        buffer = Buffer._create_buffer_from_data(data=data, shape=(2, 2), stride=(i, 1), offset=i)
        assert buffer.numel == 4

        buffer = Buffer._create_buffer_from_data(data=data, shape=(3, 2), stride=(2, 1), offset=i)
        assert buffer.numel == 6

    assert Buffer(True).numel == 1
    assert Buffer(0).numel == 1
    assert Buffer(-3.14).numel == 1


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
        buffer = Buffer._create_buffer_from_data(data=data, shape=(numel - offset,), stride=(1,), offset=offset)
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
        buffer = Buffer._create_buffer_from_data(data=data, shape=shape, stride=stride, offset=0)
        idx = 0
        for element in buffer:
            assert element == data[idx]
            idx += 1
        assert idx == numel

    # Multidimensional buffer
    numel = 24
    data = [i for i in range(numel)]
    buffer = Buffer._create_buffer_from_data(data=data, shape=(2, 2), stride=(2, 1), offset=0)
    idx = 0
    for element in buffer:
        assert element == data[idx]
        idx += 1
    assert idx == 4

    for offset in range(2):
        buffer = Buffer._create_buffer_from_data(data=data, shape=(2, 2), stride=(1, 2), offset=offset)
        idx = 0
        expected_output = [0, 2, 1, 3]
        for element in buffer:
            assert element == expected_output[idx] + offset
            idx += 1
        assert idx == 4


def test_binary_ops():
    ops = [
        lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y,
        lambda x, y: x < y, lambda x, y: x <= y, lambda x, y: x > y, lambda x, y: x >= y,
        lambda x, y: x == y, lambda x, y: x != y,
    ]
    div_op_idx = 3

    for op_idx, op in enumerate(ops):
        # Scalars
        scalars = [-3, -2, -1, 0, 1, 2, 3]
        for first_scalar in scalars:
            for second_scalar in scalars:
                for first_dtype in DType:
                    for second_dtype in DType:
                        first_buffer = Buffer(first_scalar, first_dtype)
                        second_buffer = Buffer(second_scalar, second_dtype)

                        if first_dtype != second_dtype:
                            with pytest.raises(ValueError, match="DType mismatch*"):
                                result = op(first_buffer, second_buffer)

                        elif div_op_idx == op_idx and second_scalar == 0:
                            with pytest.raises(ZeroDivisionError):
                                result = op(first_buffer, second_buffer)

                        else:
                            result = op(first_buffer, second_buffer)
                            expected_result = op(first_dtype.cast(first_scalar), second_dtype.cast(second_scalar))

                            assert result.shape == ()
                            assert result.offset == 0
                            assert result.stride == ()
                            assert result.dtype == DType.deduce_dtype(expected_result)
                            assert result.data[0] == expected_result

                            # Do it again without buffer for one of them
                            result = op(first_buffer, second_dtype.cast(second_scalar))

                            assert result.shape == ()
                            assert result.offset == 0
                            assert result.stride == ()
                            assert result.dtype == DType.deduce_dtype(expected_result)
                            assert result.data[0] == expected_result

                            result = op(first_dtype.cast(first_scalar), second_buffer)

                            assert result.shape == ()
                            assert result.offset == 0
                            assert result.stride == ()
                            assert result.dtype == DType.deduce_dtype(expected_result)
                            assert result.data[0] == expected_result

        # Tensors
        data = [[1,], [0,], [[423, 214, 5734, 434]], [[[[1], [2]], [[3], [4]]]]]
        linearized_data = [[1], [0], [423, 214, 5734, 434], [1, 2, 3, 4]]
        for idx, first_tensor in enumerate(data):
            for first_dtype in DType:
                for second_dtype in DType:
                    first_buffer = Buffer(first_tensor, first_dtype)
                    second_buffer = Buffer(first_tensor, second_dtype)

                    if first_dtype != second_dtype:
                        with pytest.raises(ValueError, match="DType mismatch*"):
                            result = op(first_buffer, second_buffer)

                    elif div_op_idx == op_idx and any(value == 0 for value in linearized_data[idx]):
                        with pytest.raises(ZeroDivisionError):
                            result = op(first_buffer, second_buffer)

                    else:
                        result = op(first_buffer, second_buffer)

                        expected_result = [
                            op(first_dtype.cast(element), second_dtype.cast(element))
                            for element in linearized_data[idx]
                        ]

                        assert result.shape == first_buffer.shape
                        assert result.offset == first_buffer.offset
                        assert result.stride == first_buffer.stride
                        assert result.dtype == DType.deduce_dtype(expected_result[0])
                        assert result.data == expected_result

        # Different shapes
        data = [0, [1,], [[423, 214, 5734, 434]], [[[[1], [2]], [[3], [4]]]]]
        for first_tensor in data:
            for second_tensor in data:
                if first_tensor != second_tensor:
                    for dtype in DType:
                        first_buffer = Buffer(first_tensor, dtype)
                        second_buffer = Buffer(second_tensor, dtype)
                        with pytest.raises(RuntimeError):
                            result = op(first_buffer, second_buffer)

        # Other type that it's not Buffer
        buffer = Buffer(1)
        for other in [(1,), [1,], (), [], None]:
            with pytest.raises(TypeError):
                result = op(buffer, other)


def test_unary_ops():
    ops = [lambda x: -x, lambda x: x ** 2]

    for op_idx, op in enumerate(ops):
        # Scalars
        scalars = [-3, -2, -1, 0, 1, 2, 3]
        for scalar in scalars:
            for dtype in DType:

                # Create a Buffer with the current scalar and dtype
                buffer = Buffer(scalar, dtype)

                # Do the operation
                result = op(buffer)
                expected_result = op(dtype.cast(scalar))

                # Check the result
                assert result.shape == ()
                assert result.offset == 0
                assert result.stride == ()
                assert result.dtype == DType.deduce_dtype(expected_result)
                assert result.data[0] == expected_result

        # Tensors
        data = [[1,], [0,], [[423, 214, 5734, 434]], [[[[1], [2]], [[3], [4]]]]]
        linearized_data = [[1], [0], [423, 214, 5734, 434], [1, 2, 3, 4]]
        for idx, tensor in enumerate(data):
            for dtype in DType:

                # Create the Buffer
                buffer = Buffer(tensor, dtype)

                # Do the operation
                result = op(buffer)

                # Check the result
                expected_result = [op(dtype.cast(element)) for element in linearized_data[idx]]

                assert result.shape == buffer.shape
                assert result.offset == buffer.offset
                assert result.stride == buffer.stride
                assert result.dtype == DType.deduce_dtype(expected_result[0])
                assert result.data == expected_result


def test_reshape():
    # Scalar reshape
    for scalar in [-3, -2, -1, 0, 1, 2, 3]:
        for dtype in DType:
            buffer = Buffer(scalar, dtype)

            # Try adding more dimensions
            new_shape = buffer.shape
            for _ in range(5):
                new_buffer = buffer.reshape(new_shape=new_shape)

                assert new_buffer.shape == new_shape
                assert id(new_buffer.data) == id(buffer.data)
                assert new_buffer.offset == 0
                assert new_buffer.ndim == len(new_shape)
                for new_element, old_element in zip(new_buffer, buffer):
                    assert new_element == old_element

                # Update new_shape for next iteration
                new_shape = (1, *new_shape, 1)

    # Tensor reshape
    for dtype in DType:
        buffer = Buffer([1, 2, 3, 4], dtype=dtype)
        for new_shape in [(4,), (2, 2), (2, 1, 1, 2)]:
            for i in range(5):
                current_new_shape = (1,) * i + new_shape + (1,) * i
                new_buffer = buffer.reshape(current_new_shape)

                assert new_buffer.shape == current_new_shape
                assert id(new_buffer.data) == id(buffer.data)
                assert new_buffer.offset == 0
                assert new_buffer.ndim == len(current_new_shape)
                for new_element, old_element in zip(new_buffer, buffer):
                    assert new_element == old_element

        buffer = Buffer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=dtype)
        for new_shape in [
            (12,), (4, 3), (3, 4), (2, 2, 3), (3, 2, 2), (2, 3, 2), (4, 1, 1, 3), (3, 1, 4), (2, 1, 2, 1, 3),
            (3, 1, 2, 2), (2, 3, 1, 2)
        ]:
            for i in range(5):
                current_new_shape = (1,) * i + new_shape + (1,) * i
                new_buffer = buffer.reshape(current_new_shape)

                assert new_buffer.shape == current_new_shape
                assert id(new_buffer.data) == id(buffer.data)
                assert new_buffer.offset == 0
                assert new_buffer.ndim == len(current_new_shape)
                for new_element, old_element in zip(new_buffer, buffer):
                    assert new_element == old_element

    # Try with non-contiguous data
    data = [i for i in range(50)]
    for offset in [0, 1, 2]:
        # 12 elements
        buffer = Buffer._create_buffer_from_data(data=data, shape=(12,), stride=(2,), offset=offset)
        new_shape = (6, 2)
        new_buffer = buffer.reshape(new_shape=new_shape)

        assert new_buffer.shape == new_shape
        assert new_buffer.ndim == len(new_shape)
        assert new_buffer.offset == 0
        assert id(new_buffer.data) != id(data)
        assert new_buffer.is_contiguous()
        for new_element, old_element in zip(new_buffer, [offset + i * 2 for i in range(12)]):
            assert new_element == old_element

        # 6 elements
        new_shape = (2, 3)
        buffer = Buffer._create_buffer_from_data(data=data, shape=(6,), stride=(4,), offset=offset)
        new_buffer = buffer.reshape(new_shape=new_shape)

        assert new_buffer.shape == new_shape
        assert new_buffer.ndim == len(new_shape)
        assert new_buffer.offset == 0
        assert id(new_buffer.data) != id(data)
        assert new_buffer.is_contiguous()
        for new_element, old_element in zip(new_buffer, [offset + i * 4 for i in range(6)]):
            assert new_element == old_element

        # 1 element
        new_shape = ()
        buffer = Buffer._create_buffer_from_data(data=data, shape=(1, 1), stride=(2, 2), offset=offset)
        new_buffer = buffer.reshape(new_shape=new_shape)

        assert new_buffer.shape == new_shape
        assert new_buffer.ndim == len(new_shape)
        assert new_buffer.offset == 0
        assert id(new_buffer.data) != id(data)
        assert new_buffer.is_contiguous()
        for new_element, old_element in zip(new_buffer, [offset]):
            assert new_element == old_element

    # Check wrong inputs
    for dtype in DType:
        buffer = Buffer([1, 2, 3, 4], dtype=dtype)

        # Wrong type
        with pytest.raises(TypeError):
            new_buffer = buffer.reshape(new_shape=None)

        # Wrong shape value
        for wrong_new_shape in [(4, -1), (-1,), (4, 0), (-2, -2)]:
            with pytest.raises(ValueError):
                new_buffer = buffer.reshape(new_shape=wrong_new_shape)

        for wrong_new_shape in [(1,), (2, 2, 1, 1, 2), (12,), (2,), (4, 3, 2, 1)]:
            with pytest.raises(RuntimeError):
                new_buffer = buffer.reshape(new_shape=wrong_new_shape)


def test_expand():
    # Scalar expansion
    for scalar in [-3, -2, -1, 0, 1, 2, 3]:
        for dtype in DType:
            buffer = Buffer(scalar, dtype)
            new_buffer = buffer.expand(new_shape=())
            assert new_buffer.shape == ()
            assert id(new_buffer.data) == id(buffer.data)
            assert new_buffer.offset == 0
            assert new_buffer.ndim == 0
            assert new_buffer.is_contiguous()
            for new_element, old_element in zip(new_buffer, buffer):
                assert new_element == old_element

    # Tensor expansion
    for dtype in DType:
        buffer = Buffer([1, 2, 3, 4], dtype=dtype)

        new_shape = (4,)
        new_buffer = buffer.expand(new_shape)

        assert new_buffer.shape == new_shape
        assert id(new_buffer.data) == id(buffer.data)
        assert new_buffer.offset == 0
        assert new_buffer.ndim == len(new_shape)
        assert new_buffer.is_contiguous()
        for new_element, old_element in zip(new_buffer, buffer):
            assert new_element == old_element

        buffer = Buffer._create_buffer_from_data(
            data=buffer.data, shape=(4, 1), stride=Buffer._calculate_stride((4, 1)), offset=0
        )
        for i in range(5):
            new_shape = (4, i + 1)
            new_buffer = buffer.expand(new_shape)

            assert new_buffer.shape == new_shape
            assert id(new_buffer.data) == id(buffer.data)
            assert new_buffer.offset == 0
            assert new_buffer.ndim == len(new_shape)
            assert new_buffer.is_contiguous() if i == 0 else not new_buffer.is_contiguous()
            expected_data = [*(1,) * (i + 1), *(2,) * (i + 1), *(3,) * (i + 1), *(4,) * (i + 1)]
            for idx, new_element in enumerate(new_buffer):
                assert new_element == dtype.cast(expected_data[idx])

        buffer = Buffer._create_buffer_from_data(
            data=buffer.data, shape=(1, 4), stride=Buffer._calculate_stride((1, 4)), offset=0
        )
        for i in range(5):
            new_shape = (i + 1, 4)
            new_buffer = buffer.expand(new_shape)

            assert new_buffer.shape == new_shape
            assert id(new_buffer.data) == id(buffer.data)
            assert new_buffer.offset == 0
            assert new_buffer.ndim == len(new_shape)
            assert new_buffer.is_contiguous() if i == 0 else not new_buffer.is_contiguous()
            for idx, new_element in enumerate(new_buffer):
                new_element == dtype.cast(idx % 4 + 1)

        data = [dtype.cast(e) for e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        for shape in [(3, 4, 1), (4, 3, 1)]:
            buffer = Buffer._create_buffer_from_data(
                data=data, shape=shape, stride=Buffer._calculate_stride(shape), offset=0
            )
            for i in range(5):
                new_shape = (*shape[:-1], 1 + i)
                new_buffer = buffer.expand(new_shape)

                assert new_buffer.shape == new_shape
                assert id(new_buffer.data) == id(buffer.data)
                assert new_buffer.offset == 0
                assert new_buffer.ndim == len(new_shape)
                assert new_buffer.is_contiguous() if i == 0 else not new_buffer.is_contiguous()
                expected_data_idx = -1
                for idx, new_element in enumerate(new_buffer):
                    if (i + 1) < 1 or idx % (i + 1) == 0:
                        expected_data_idx += 1
                    assert new_element == dtype.cast(data[expected_data_idx])

    # Try with non-contiguous data
    data = [i for i in range(50)]
    for offset in [0, 1, 2]:
        buffer = Buffer._create_buffer_from_data(data=data, shape=(6, 1), stride=(2, 1), offset=offset)
        expected_data = [offset + i * 2 for i in range(6)]
        for i in range(3):
            new_shape = (6, i + 1)
            new_buffer = buffer.expand(new_shape=new_shape)

            assert new_buffer.shape == new_shape
            assert new_buffer.ndim == len(new_shape)
            assert new_buffer.offset == 0
            assert id(new_buffer.data) != id(data)
            assert new_buffer.is_contiguous() if i == 0 else not new_buffer.is_contiguous()

            expected_data_idx = -1
            for idx, new_element in enumerate(new_buffer):
                if (i + 1) < 1 or idx % (i + 1) == 0:
                    expected_data_idx += 1
                assert new_element == expected_data[expected_data_idx]

    # Check wrong inputs
    for dtype in DType:
        buffer = Buffer([[1, 2, 3, 4]], dtype=dtype)

        # Wrong type
        with pytest.raises(TypeError):
            new_buffer = buffer.expand(new_shape=None)

        # Wrong shape value
        for wrong_new_shape in [(4, -1), (-1,), (4, 0), (-2, -2)]:
            with pytest.raises(ValueError):
                new_buffer = buffer.expand(new_shape=wrong_new_shape)

        for wrong_new_shape in [(1,), (2, 2, 1, 1, 2), (12,), (2,), (4, 3, 2, 1)]:
            with pytest.raises(ValueError):
                new_buffer = buffer.expand(new_shape=wrong_new_shape)


def test_generate_indexes():
    # Scalar
    buffer = Buffer(1.0)
    indexes = [index for index in buffer._generate_indexes()]

    assert indexes == [(0,)]

    # 1D
    buffer = Buffer([i * 7 for i in range(10)])
    indexes = [index for index in buffer._generate_indexes()]

    assert indexes == [(i,) for i in range(10)]

    # 2D
    buffer = Buffer([[i * 2, j * 5] for i in range(2) for j in range(10)])
    indexes = [index for index in buffer._generate_indexes()]

    expected_indexes = []
    for i in range(20):
        for j in range(2):
            expected_indexes.append((i, j))
    assert indexes == expected_indexes

    # 3D
    buffer = Buffer([[[i * 2, i * 5, i], [j * 2, j * 5, j]] for i in range(2) for j in range(10)])
    indexes = [index for index in buffer._generate_indexes()]

    expected_indexes = []
    for i in range(20):
        for j in range(2):
            for k in range(3):
                expected_indexes.append((i, j, k))

    assert indexes == expected_indexes


def test_reduce_sum():
    # Scalar
    buffer = Buffer(1.0)

    with pytest.raises(ValueError):
        result = buffer.sum(0)

    # 1D
    buffer = Buffer([i for i in range(9)])

    result = buffer.sum(0)
    assert all(result == Buffer([sum([i for i in range(9)])]))

    # 2D
    buffer = Buffer([[1, 2, 3], [4, 5, 6]])

    result = buffer.sum(0)
    assert all(result == Buffer([[5, 7, 9]]))

    result = buffer.sum(1)
    assert all(result == Buffer([[6], [15]]))

    # 3D
    buffer = Buffer([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    result = buffer.sum(0)
    assert all(result == Buffer([[[6, 8], [10, 12]]]))

    result = buffer.sum(1)
    assert all(result == Buffer([[[4, 6]], [[12, 14]]]))

    result = buffer.sum(2)
    assert all(result == Buffer([[[3], [7]], [[11], [15]]]))

    # Wrong inputs
    with pytest.raises(ValueError):
        _ = buffer.sum(4)

    with pytest.raises(ValueError):
        _ = buffer.sum(-1)


def test_reduce_max():
    # Scalar
    buffer = Buffer(1.0)

    with pytest.raises(ValueError):
        result = buffer.max(0)

    # 1D
    buffer = Buffer([i for i in range(9)])

    result = buffer.max(0)
    assert all(result == Buffer([max([i for i in range(9)])]))

    # 2D
    buffer = Buffer([[1, 2, 3], [4, 5, 6]])

    result = buffer.max(0)
    assert all(result == Buffer([[4, 5, 6]]))

    result = buffer.max(1)
    assert all(result == Buffer([[3], [6]]))

    # 3D
    buffer = Buffer([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    result = buffer.max(0)
    assert all(result == Buffer([[[5, 6], [7, 8]]]))

    result = buffer.max(1)
    assert all(result == Buffer([[[3, 4]], [[7, 8]]]))

    result = buffer.max(2)
    assert all(result == Buffer([[[2], [4]], [[6], [8]]]))

    # Wrong inputs
    with pytest.raises(ValueError):
        _ = buffer.max(4)

    with pytest.raises(ValueError):
        _ = buffer.max(-1)


def test_pow():
    # Scalar
    buffer = Buffer(3.0)
    assert all((buffer ** 3.0) == Buffer(3.0 ** 3.0))

    # 1D Tensor
    data = [3, 5]
    buffer = Buffer(data)
    assert all((buffer ** 3.0) == Buffer([e ** 3.0 for e in data]))
    assert all((buffer ** 2) == Buffer([e ** 2 for e in data]))

    # 2D Tensor
    data = [3, 5]
    buffer = Buffer([data, data])
    assert all((buffer ** 3.0) == Buffer([[e ** 3.0 for e in data], [e ** 3.0 for e in data]]))
    assert all((buffer ** 2) == Buffer([[e ** 2 for e in data], [e ** 2 for e in data]]))

    # 3D Tensor
    data = [3, 5, 5]
    buffer = Buffer([[data], [data]])
    assert all((buffer ** 3.0) == Buffer([[[e ** 3.0 for e in data]], [[e ** 3.0 for e in data]]]))
    assert all((buffer ** 2) == Buffer([[[e ** 2 for e in data]], [[e ** 2 for e in data]]]))

    # Wrong exponent type
    buffer = Buffer([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(TypeError):
        _ = buffer ** buffer

    with pytest.raises(TypeError):
        _ = buffer ** Buffer(1.0)


def test_exp():
    # Scalar
    buffer = Buffer(3.0)
    assert all((buffer.exp()) == Buffer(math.exp(3.0)))

    # 1D Tensor
    data = [3, 5]
    buffer = Buffer(data)
    assert all((buffer.exp()) == Buffer([math.exp(e) for e in data]))

    # 2D Tensor
    data = [3, 5]
    buffer = Buffer([data, data])
    assert all((buffer.exp()) == Buffer([[math.exp(e) for e in data], [math.exp(e) for e in data]]))

    # 3D Tensor
    data = [3, 5, 5]
    buffer = Buffer([[data], [data]])
    assert all((buffer.exp()) == Buffer([[[math.exp(e) for e in data]], [[math.exp(e) for e in data]]]))


def test_log():
    # Scalar
    buffer = Buffer(3.0)
    assert all((buffer.log()) == Buffer(math.log(3.0)))

    # 1D Tensor
    data = [3, 5]
    buffer = Buffer(data)
    assert all((buffer.log()) == Buffer([math.log(e) for e in data]))

    # 2D Tensor
    data = [3, 5]
    buffer = Buffer([data, data])
    assert all((buffer.log()) == Buffer([[math.log(e) for e in data], [math.log(e) for e in data]]))

    # 3D Tensor
    data = [3, 5, 5]
    buffer = Buffer([[data], [data]])
    assert all((buffer.log()) == Buffer([[[math.log(e) for e in data]], [[math.log(e) for e in data]]]))


def test_maximum():
    # Scalar
    first_buffer = Buffer(3.0)
    second_buffer = Buffer(5.0)
    assert all((first_buffer.maximum(second_buffer)) == Buffer(5.0))
    assert all((second_buffer.maximum(first_buffer)) == Buffer(5.0))

    # 1D Tensor
    first_buffer = Buffer([3, 5])
    second_buffer = Buffer([6, 2])
    assert all((first_buffer.maximum(second_buffer)) == Buffer([6, 5]))
    assert all((second_buffer.maximum(first_buffer)) == Buffer([6, 5]))

    # 2D Tensor
    first_buffer = Buffer([[-3, 1], [4, 5]])
    second_buffer = Buffer([[0, 2], [3, 5]])
    assert all((first_buffer.maximum(second_buffer)) == Buffer([[0, 2], [4, 5]]))
    assert all((second_buffer.maximum(first_buffer)) == Buffer([[0, 2], [4, 5]]))

    # 3D Tensor
    first_buffer = Buffer([[[3, 2, 5]], [[1, 1, 1]]])
    second_buffer = Buffer([[[1, 5, 15]], [[3, 5, 5]]])
    assert all((first_buffer.maximum(second_buffer)) == Buffer([[[3, 5, 15]], [[3, 5, 5]]]))
    assert all((second_buffer.maximum(first_buffer)) == Buffer([[[3, 5, 15]], [[3, 5, 5]]]))


def test_float():
    for dtype in DType:
        # Scalar
        buffer = Buffer(dtype.cast(0))
        assert all((buffer.float()) == Buffer(0.0))

        # 1D Tensor
        data = [dtype.cast(0), dtype.cast(13)]
        buffer = Buffer(data)

        if dtype == DType.bool:
            assert all((buffer.float()) == Buffer([0., 1.]))
        else:
            assert all((buffer.float()) == Buffer([0., 13.]))

        # 2D Tensor
        data = [dtype.cast(3), dtype.cast(5), dtype.cast(0)]
        buffer = Buffer([data, data])

        if dtype == DType.bool:
            assert all((buffer.float()) == Buffer([[1., 1., 0.], [1., 1., 0.]]))
        else:
            assert all((buffer.float()) == Buffer([[3., 5., 0.], [3., 5., 0.]]))

        # 3D Tensor
        data = [dtype.cast(3), dtype.cast(0), dtype.cast(5)]
        buffer = Buffer([[data], [data]])

        if dtype == DType.bool:
            assert all((buffer.float()) == Buffer([[[1., 0., 1.]], [[1., 0., 1.]]]))
        else:
            assert all((buffer.float()) == Buffer([[[3., 0., 5.]], [[3., 0., 5.]]]))

    # Bool operation to float
    first_buffer = Buffer([[1.2, 0.8, 0.], [1., -1., 0.]])
    second_buffer = Buffer([[1., 1., 0.], [12., 0., 2.]])

    assert all((first_buffer > second_buffer).float() == Buffer([[1., 0., 0.], [0., 0., 0.]]))


def test_init_methods():
    # Wrong arguments
    with pytest.raises(TypeError):
        Buffer.uniform(None)

    with pytest.raises(TypeError):
        Buffer.uniform((0))

    with pytest.raises(TypeError):
        Buffer.uniform([])

    with pytest.raises(ValueError):
        Buffer.uniform((-1,))

    with pytest.raises(ValueError):
        Buffer.uniform((1, 2, 3, 0))

    with pytest.raises(ValueError):
        Buffer.uniform((1, -1, 3))

    # Empty buffer
    buffer = Buffer.uniform(())
    assert buffer.shape == ()
    assert buffer.dtype == DType.float32

    # 1D
    buffer = Buffer.uniform((54,))
    assert buffer.shape == (54,)
    assert buffer.dtype == DType.float32
    assert all(buffer >= 0)
    assert all(buffer <= 1.0)

    # 2D
    buffer = Buffer.uniform((16, 32))
    assert buffer.shape == (16, 32)
    assert buffer.dtype == DType.float32
    assert all(buffer >= 0)
    assert all(buffer <= 1.0)

    # 3D
    buffer = Buffer.uniform((12, 13, 7))
    assert buffer.shape == (12, 13, 7)
    assert buffer.dtype == DType.float32
    assert all(buffer >= 0)
    assert all(buffer <= 1.0)


def test_permute():

    # Wrong values
    buffer = Buffer([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(TypeError):
        buffer.permute(None)

    with pytest.raises(ValueError):
        buffer.permute(())

    with pytest.raises(ValueError):
        buffer.permute((2,))

    with pytest.raises(ValueError):
        buffer.permute((0, 1, 2))

    with pytest.raises(ValueError):
        buffer.permute((0, 3))

    with pytest.raises(ValueError):
        buffer.permute((0, 0))

    # Scalar
    buffer = Buffer(3.0)
    assert all(buffer.permute(()) == buffer)

    # 1D Tensor
    buffer = Buffer([3, 5])
    assert all(buffer.permute((0,)) == buffer)

    # 2D Tensor
    buffer = Buffer([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert all(buffer.permute((0, 1)) == buffer)
    assert all(buffer.permute((1, 0)) == Buffer([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))

    # 3D Tensor
    buffer = Buffer(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]
        ]
    )

    assert all(buffer.permute((0, 1, 2)) == buffer)

    assert all(buffer.permute((0, 2, 1)) == Buffer(
            [
                [[1., 4.], [2., 5.], [3., 6.]],
                [[7., 10.], [8., 11.], [9., 12.]],
                [[13., 16.], [14., 17.], [15., 18.]],
                [[19., 22.], [20., 23.], [21., 24.]]
            ]
        )
    )

    assert all(buffer.permute((1, 0, 2)) == Buffer(
            [
                [[1., 2., 3.], [7., 8., 9.], [13., 14., 15.], [19., 20., 21.]],
                [[4., 5., 6.], [10., 11., 12.], [16., 17., 18.], [22., 23., 24.]]
            ]
        )
    )

    assert all(buffer.permute((1, 2, 0)) == Buffer(
            [
                [[1., 7., 13., 19.], [2., 8., 14., 20.], [3., 9., 15., 21.]],
                [[4., 10., 16., 22.], [5., 11., 17., 23.], [6., 12., 18., 24.]]
            ]
        )
    )

    assert all(buffer.permute((2, 1, 0)) == Buffer(
            [
                [[1., 7., 13., 19.], [4., 10., 16., 22.]],
                [[2., 8., 14., 20.], [5., 11., 17., 23.]],
                [[3., 9., 15., 21.], [6., 12., 18., 24.]]
            ]
        )
    )

    assert all(buffer.permute((2, 0, 1)) == Buffer(
            [
                [[1., 4.], [7., 10.], [13., 16.], [19., 22.]],
                [[2., 5.], [8., 11.], [14., 17.], [20., 23.]],
                [[3., 6.], [9., 12.], [15., 18.], [21., 24.]]
            ]
        )
    )

    # 1-D non-contiguous data
    buffer = Buffer._create_buffer_from_data(data=[i for i in range(24)], shape=(4,), stride=(4,), offset=0)
    assert all(buffer.permute((0,)) == buffer)

    # 2-D non-contiguous data
    buffer = Buffer._create_buffer_from_data(data=[i for i in range(24)], shape=(3, 2), stride=(4, 1), offset=0)
    assert all(buffer.permute((0, 1)) == Buffer([[0, 1], [4, 5], [8, 9]]))
    assert all(buffer.permute((1, 0)) == Buffer([[0, 4, 8], [1, 5, 9]]))

    # 3-D non-contiguous data
    buffer = Buffer._create_buffer_from_data(data=[i for i in range(64)], shape=(3, 2, 2), stride=(4, 4, 1), offset=0)

    assert all(buffer.permute((0, 1, 2)) == buffer)
    assert all(buffer.permute((0, 2, 1)) == Buffer([[[0, 4], [1, 5]], [[4, 8], [5, 9]], [[8, 12], [9, 13]]]))
    assert all(buffer.permute((1, 0, 2)) == Buffer([[[0, 1], [4, 5], [8, 9]], [[4, 5], [8, 9], [12, 13]]]))
    assert all(buffer.permute((1, 2, 0)) == Buffer([[[0, 4, 8], [1, 5, 9]], [[4, 8, 12], [5, 9, 13]]]))
    assert all(buffer.permute((2, 1, 0)) == Buffer([[[0, 4, 8], [4, 8, 12]], [[1, 5, 9], [5, 9, 13]]]))
    assert all(buffer.permute((2, 0, 1)) == Buffer([[[0, 4], [4, 8], [8, 12]], [[1, 5], [5, 9], [9, 13]]]))


def test_to_python():

    for dtype in DType:
        for value in [dtype.cast(-1), dtype.cast(0), dtype.cast(1)]:
            for data in [value, [value], [], [value, value, value], [[value, value], [value, value]], [[], []]]:
                # Create a Buffer
                buffer = Buffer(data)

                # Get the Python value
                python_data = buffer.to_python()

                # Check is the expected value
                assert isinstance(python_data, type(data))
                assert python_data == data