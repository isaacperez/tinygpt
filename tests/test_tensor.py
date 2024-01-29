import math

import pytest

from tinygpt.tensor import Tensor
from tinygpt.buffer import Buffer
from tinygpt.utils import DType


def test_Tensor():
    # Testing creation of an empty tensor
    tensor = Tensor([])
    assert tensor.shape == (0,)
    assert tensor.ndim == 1
    assert tensor.dtype == DType.float32
    assert not tensor.requires_grad

    # Testing tensor creation with scalar values
    for scalar in [1.0, -1.0, 0.0, -1, 0, 1, True, False]:
        tensor = Tensor(scalar)
        assert tensor.shape == ()
        assert tensor.ndim == 0
        assert tensor.dtype == DType.deduce_dtype(scalar)
        assert not tensor.requires_grad

    # Testing tensor creation with 1D list
    tensor = Tensor([1, 2, 3])
    assert tensor.shape == (3,)
    assert tensor.ndim == 1
    assert tensor.dtype == DType.int32
    assert not tensor.requires_grad

    # Testing tensor creation with 2D list
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    assert tensor.shape == (2, 3)
    assert tensor.ndim == 2
    assert tensor.dtype == DType.int32
    assert not tensor.requires_grad

    # Testing tensor creation with 3D list
    tensor = Tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert tensor.shape == (3, 2, 2)
    assert tensor.ndim == 3
    assert tensor.dtype == DType.int32
    assert not tensor.requires_grad

    # Testing tensor with requires_grad set to True
    tensor = Tensor([], requires_grad=True)
    assert tensor.shape == (0,)
    assert tensor.ndim == 1
    assert tensor.dtype == DType.float32
    assert tensor.requires_grad

    for scalar in [-1.0, 0.0, 1.0]:
        tensor = Tensor(scalar, requires_grad=True)
        assert tensor.shape == ()
        assert tensor.ndim == 0
        assert tensor.dtype == DType.deduce_dtype(scalar)
        assert tensor.requires_grad

    tensor = Tensor([[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], requires_grad=True)
    assert tensor.shape == (3, 2, 2)
    assert tensor.ndim == 3
    assert tensor.dtype == DType.float32
    assert tensor.requires_grad

    # Testing invalid dtype for requires_grad=True
    for data in ([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]], 1, True, False, [1, 2], [True, False]):
        with pytest.raises(RuntimeError, match="Only float32 Tensors can require gradients"):
            tensor = Tensor(data, requires_grad=True)


def test_add():
    # Test addition of two tensors
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 + tensor2
    assert result.shape == (3,)
    assert not result.requires_grad
    assert all(result.buffer == Buffer([5, 7, 9]))

    # Test addition of 3D tensors
    tensor1 = Tensor([[[1.], [2.]], [[3.], [4.]]], requires_grad=True)
    tensor2 = Tensor([[[5.], [6.]], [[7.], [8.]]])
    result = tensor1 + tensor2
    assert result.shape == (2, 2, 1)
    assert result.requires_grad
    assert all(result.buffer == Buffer([[[6.], [8.]], [[10.], [12.]]]))


def test_sub():
    # Test subtraction of two tensors
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 - tensor2
    assert result.shape == (3,)
    assert not result.requires_grad
    assert all(result.buffer == Buffer([-3, -3, -3]))

    # Test subtraction of 3D tensors
    tensor1 = Tensor([[[1.], [2.]], [[3.], [4.]]], requires_grad=True)
    tensor2 = Tensor([[[5.], [6.]], [[7.], [8.]]])
    result = tensor1 - tensor2
    assert result.shape == (2, 2, 1)
    assert result.requires_grad
    assert all(result.buffer == Buffer([[[-4.], [-4.]], [[-4.], [-4.]]]))


def test_mul():
    # Test multiplication of two tensors
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 * tensor2
    assert result.shape == (3,)
    assert not result.requires_grad
    assert all(result.buffer == Buffer([4, 10, 18]))

    # Test multiplication of 3D tensors
    tensor1 = Tensor([[[1.], [2.]], [[3.], [4.]]], requires_grad=True)
    tensor2 = Tensor([[[5.], [6.]], [[7.], [8.]]])
    result = tensor1 * tensor2
    assert result.shape == (2, 2, 1)
    assert result.requires_grad
    assert all(result.buffer == Buffer([[[5.], [12.]], [[21.], [32.]]]))


def test_div():
    # Test division of two tensors
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 / tensor2
    assert result.shape == (3,)
    assert not result.requires_grad
    assert all(result.buffer == Buffer([1 / 4, 2 / 5, 3 / 6]))

    # Test division of 3D tensors
    tensor1 = Tensor([[[1.], [2.]], [[3.], [4.]]], requires_grad=True)
    tensor2 = Tensor([[[5.], [6.]], [[7.], [8.]]])
    result = tensor1 / tensor2
    assert result.shape == (2, 2, 1)
    assert result.requires_grad
    assert all(result.buffer == Buffer([[[1. / 5.], [2. / 6.]], [[3. / 7.], [4. / 8.]]]))


def test_backward_non_scalar_tensor():
    # Test backward method on non-scalar tensor
    tensor = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
    with pytest.raises(RuntimeError):
        tensor.backward(incoming_gradient=None)

    with pytest.raises(TypeError):
        tensor.backward(incoming_gradient=1)

    with pytest.raises(TypeError):
        tensor.backward(incoming_gradient=1.0)


def test_gradient_accumulation():
    # Test if gradients are accumulated correctly
    tensor = Tensor(5.0, requires_grad=True)
    tensor.backward(Buffer(1.0))
    tensor.backward(Buffer(2.0))
    assert all(tensor.grad == Buffer(3.0))


def test_dtype_consistency_operations():
    # Test operation between different dtypes
    tensor1 = Tensor([1, 2, 3], dtype=DType.int32)
    tensor2 = Tensor([1.0, 2.0, 3.0], dtype=DType.float32)
    with pytest.raises(ValueError):
        tensor1 + tensor2


def test_gradient_function_backward_with_add():
    # Test backward propagation in GradientFunction with add
    for requires_grad_tensor1, requires_grad_tensor2 in [(True, True), (True, False), (False, True)]:
        tensor1 = Tensor(1.0, requires_grad=requires_grad_tensor1)
        tensor2 = Tensor(2.0, requires_grad=requires_grad_tensor2)

        assert tensor1.grad is None
        assert tensor2.grad is None

        result = tensor1 + tensor2

        assert result.grad is None
        assert result.grad_fn is not None

        result.retain_grad()
        result.backward()

        if requires_grad_tensor1:
            assert all(tensor1.grad == Buffer(1.0))
        else:
            assert tensor1.grad is None

        if requires_grad_tensor2:
            assert all(tensor2.grad == Buffer(1.0))
        else:
            assert tensor2.grad is None

        assert all(result.grad == Buffer(1.0))
        assert result.grad_fn is None


def test_gradient_function_backward_with_sub():
    # Test backward propagation in GradientFunction with sub
    for requires_grad_tensor1, requires_grad_tensor2 in [(True, True), (True, False), (False, True)]:
        tensor1 = Tensor(1.0, requires_grad=requires_grad_tensor1)
        tensor2 = Tensor(2.0, requires_grad=requires_grad_tensor2)

        assert tensor1.grad is None
        assert tensor2.grad is None

        result = tensor1 - tensor2

        assert result.grad is None
        assert result.grad_fn is not None
        
        result.retain_grad()
        result.backward()

        if requires_grad_tensor1:
            assert all(tensor1.grad == Buffer(1.0))
        else:
            assert tensor1.grad is None

        if requires_grad_tensor2:
            assert all(tensor2.grad == Buffer(-1.0))
        else:
            assert tensor2.grad is None

        assert all(result.grad == Buffer(1.0))
        assert result.grad_fn is None


def test_gradient_function_backward_with_neg():
    # Test backward propagation in GradientFunction with neg
    for requires_grad in [True, False]:
        tensor = Tensor(1.0, requires_grad=requires_grad)

        assert tensor.grad is None

        result = -tensor

        assert result.grad is None
        if requires_grad:
            assert result.grad_fn is not None
        else:
            assert result.grad_fn is None

        result.retain_grad()
        result.backward()

        if requires_grad:
            assert all(tensor.grad == Buffer(-1.0))
            assert all(result.grad == Buffer(1.0))
            assert result.grad_fn is None

        else:
            assert tensor.grad is None
            assert result.grad is None

            assert result.grad_fn is None


def test_gradient_function_backward_with_mul():
    # Test backward propagation in GradientFunction with multiplication
    for requires_grad_tensor1, requires_grad_tensor2 in [(True, True), (True, False), (False, True)]:
        tensor1 = Tensor(2.0, requires_grad=requires_grad_tensor1)
        tensor2 = Tensor(3.0, requires_grad=requires_grad_tensor2)

        assert tensor1.grad is None
        assert tensor2.grad is None

        result = tensor1 * tensor2

        assert result.grad is None
        assert result.grad_fn is not None
        
        result.retain_grad()
        result.backward()

        if requires_grad_tensor1:
            assert all(tensor1.grad == Buffer(3.0))
        else:
            assert tensor1.grad is None

        if requires_grad_tensor2:
            assert all(tensor2.grad == Buffer(2.0))
        else:
            assert tensor2.grad is None

        assert all(result.grad == Buffer(1.0))
        assert result.grad_fn is None


def test_gradient_function_backward_with_div():
    # Test backward propagation in GradientFunction with division
    for requires_grad_tensor1, requires_grad_tensor2 in [(True, True), (True, False), (False, True)]:
        tensor1 = Tensor(2.0, requires_grad=requires_grad_tensor1)
        tensor2 = Tensor(3.0, requires_grad=requires_grad_tensor2)

        assert tensor1.grad is None
        assert tensor2.grad is None

        result = tensor1 / tensor2

        assert result.grad is None
        assert result.grad_fn is not None

        result.retain_grad()
        result.backward()

        if requires_grad_tensor1:
            assert all(tensor1.grad == Buffer(1.0 / 3.0))
        else:
            assert tensor1.grad is None

        if requires_grad_tensor2:
            assert all(tensor2.grad == Buffer(-(2.0 / 3 ** 2)))
        else:
            assert tensor2.grad is None

        assert all(result.grad == Buffer(1.0))
        assert result.grad_fn is None


def test_gradient_function_backward_with_pow():
    # Test backward propagation in GradientFunction with pow
    for requires_grad in [True, False]:
        tensor = Tensor(3.0, requires_grad=requires_grad)

        assert tensor.grad is None

        result = tensor ** 2

        assert result.grad is None
        if requires_grad:
            assert result.grad_fn is not None
        else:
            assert result.grad_fn is None

        result.retain_grad()
        result.backward()

        if requires_grad:
            assert all(tensor.grad == Buffer(6.0))
            assert all(result.grad == Buffer(1.0))

            assert result.grad_fn is None

        else:
            assert tensor.grad is None
            assert result.grad is None

            assert result.grad_fn is None


def test_gradient_function_backward_with_exp():
    # Test backward propagation in GradientFunction with exp
    for requires_grad in [True, False]:
        tensor = Tensor(3.0, requires_grad=requires_grad)

        assert tensor.grad is None

        result = tensor.exp()

        assert result.grad is None
        if requires_grad:
            assert result.grad_fn is not None
        else:
            assert result.grad_fn is None

        result.retain_grad()
        result.backward()

        if requires_grad:
            assert all(tensor.grad == Buffer(math.exp(3.0)))
            assert all(result.grad == Buffer(1.0))

            assert result.grad_fn is None

        else:
            assert tensor.grad is None
            assert result.grad is None

            assert result.grad_fn is None


def test_gradient_function_backward_with_log():
    # Test backward propagation in GradientFunction with log
    for requires_grad in [True, False]:
        tensor = Tensor(3.0, requires_grad=requires_grad)

        assert tensor.grad is None

        result = tensor.log()

        assert result.grad is None
        if requires_grad:
            assert result.grad_fn is not None
        else:
            assert result.grad_fn is None

        result.retain_grad()
        result.backward()

        if requires_grad:
            assert all(tensor.grad == Buffer(1 / 3.0))
            assert all(result.grad == Buffer(1.0))

            assert result.grad_fn is None

        else:
            assert tensor.grad is None
            assert result.grad is None

            assert result.grad_fn is None


def test_gradient_function_backward_with_maximum():
    # Test backward propagation in GradientFunction with maximum
    for requires_grad in [True, False]:
        first_tensor = Tensor(3.0, requires_grad=requires_grad)
        second_tensor = Tensor(9.0, requires_grad=requires_grad)

        assert first_tensor.grad is None
        assert second_tensor.grad is None

        result = first_tensor.maximum(second_tensor)

        assert result.grad is None
        if requires_grad:
            assert result.grad_fn is not None
        else:
            assert result.grad_fn is None

        result.retain_grad()
        result.backward()

        if requires_grad:
            assert all(first_tensor.grad == Buffer(0.0))
            assert all(second_tensor.grad == Buffer(1.0))
            assert all(result.grad == Buffer(1.0))
            
            assert result.grad_fn is None

        else:
            assert first_tensor.grad is None
            assert second_tensor.grad is None
            assert result.grad is None

            assert result.grad_fn is None


def test_gradient_function_backward_with_relu():
    # Test backward propagation in GradientFunction with relu
    for data in [-3.0, 3.0]:
        for requires_grad in [True, False]:
            tensor = Tensor(data, requires_grad=requires_grad)

            assert tensor.grad is None

            result = tensor.relu()

            assert result.grad is None
            if requires_grad:
                assert result.grad_fn is not None
            else:
                assert result.grad_fn is None
            
            result.retain_grad()
            result.backward()

            if requires_grad:
                assert all(tensor.grad == Buffer(float(data > 0.0)))
                assert all(result.grad == Buffer(1.0))

                assert result.grad_fn is None

            else:
                assert tensor.grad is None
                assert result.grad is None

                assert result.grad_fn is None


def test_exp():
    # Test exp operation
    data = [3, 5, 5]
    tensor = Tensor([[data], [data]])
    assert all(tensor.exp().buffer == Buffer([[[math.exp(e) for e in data]], [[math.exp(e) for e in data]]]))


def test_log():
    # Test log operation
    data = [3, 5, 5]
    tensor = Tensor([[data], [data]])
    assert all(tensor.log().buffer == Buffer([[[math.log(e) for e in data]], [[math.log(e) for e in data]]]))


def test_maximum():
    # Test maximum operation
    first_tensor = Tensor([[[3, 2, 5]], [[1, 1, 1]]])
    second_tensor = Tensor([[[1, 5, 15]], [[3, 5, 5]]])
    assert all(first_tensor.maximum(second_tensor).buffer == Buffer([[[3, 5, 15]], [[3, 5, 5]]]))
    assert all(second_tensor.maximum(first_tensor).buffer == Buffer([[[3, 5, 15]], [[3, 5, 5]]]))


def test_relu():
    # Test relu operation
    data = [3, 5, 5]
    tensor = Tensor([[data], [data]])
    assert all(tensor.log().buffer == Buffer([[[math.log(e) for e in data]], [[math.log(e) for e in data]]]))


def test_multiple_ops():
    # Test multiple operations
    tensor1 = Tensor(3.0, requires_grad=True)
    tensor2 = Tensor(7.0, requires_grad=True)
    tensor3 = Tensor(9.0, requires_grad=True)
    tensor4 = Tensor(11.0, requires_grad=True)

    assert tensor1.grad is None
    assert tensor2.grad is None
    assert tensor3.grad is None
    assert tensor4.grad is None

    result1 = tensor1 + tensor2
    result2 = tensor3 + tensor4
    result3 = result1 * result2

    assert result1.grad is None
    assert result1.grad_fn is not None
    assert result2.grad is None
    assert result2.grad_fn is not None
    assert result3.grad is None
    assert result3.grad_fn is not None

    result1.retain_grad()
    result2.retain_grad()
    result3.retain_grad()
    result3.backward()

    assert all(tensor1.grad == Buffer(20.0))
    assert all(tensor2.grad == Buffer(20.0))
    assert all(tensor3.grad == Buffer(10.0))
    assert all(tensor4.grad == Buffer(10.0))

    assert all(result1.grad == Buffer(20.0))
    assert result1.grad_fn is None
    assert all(result2.grad == Buffer(10.0))
    assert result2.grad_fn is None
    assert all(result3.grad == Buffer(1.0))
    assert result3.grad_fn is None


def test_sum():
    tensor = Tensor([[1., 2.], [3., 4.]], requires_grad=True)

    new_tensor = tensor.sum((0, 1), keepdim=True)
    assert new_tensor.shape == (1, 1)
    assert all(new_tensor.buffer == Buffer([[10.0]]))

    new_tensor = tensor.sum((0, 1), keepdim=False)
    assert new_tensor.shape == ()
    assert all(new_tensor.buffer == Buffer(10.0))

    new_tensor.backward()

    assert all(tensor.grad == Buffer([[1.0, 1.0], [1.0, 1.0]]))


def test_max():
    tensor = Tensor([[1., 2.], [3., 4.]], requires_grad=True)

    new_tensor = tensor.max((0, 1), keepdim=True)
    assert new_tensor.shape == (1, 1)
    assert all(new_tensor.buffer == Buffer([[4.0]]))

    new_tensor = tensor.max((0, 1), keepdim=False)
    assert new_tensor.shape == ()
    assert all(new_tensor.buffer == Buffer(4.0))

    new_tensor.backward()

    assert all(tensor.grad == Buffer([[0.0, 0.0], [0.0, 1.0]]))


def test_reshape():
    # Multiple reshape operations
    tensor = Tensor([[[9.0]]], requires_grad=True)

    new_tensor_1 = tensor.reshape((1, 1, 1, 1, 1, 1))
    assert new_tensor_1.shape == (1, 1, 1, 1, 1, 1)

    new_tensor_2 = new_tensor_1.reshape(())
    assert new_tensor_2.shape == ()

    new_tensor_2.backward()
    assert all(tensor.grad == Buffer([[[1.0]]]))

    # Reshape with the same shape
    tensor = Tensor(7.0, requires_grad=True)

    new_tensor = tensor.reshape(())
    new_tensor.backward()

    assert all(tensor.grad == Buffer(1.0))


def test_expand():
    # Expand a scalar tensor
    tensor = Tensor(7.0, requires_grad=True)

    new_tensor = tensor.expand(())
    new_tensor.backward()

    assert all(tensor.grad == Buffer(1.0))


def test_multiple_ops_with_reduction_ops():
    # Test multiple operations using reduction ops too
    tensor = Tensor([3., 5., 7.], requires_grad=True)

    reshaped_tensor = tensor.reshape((1, 3))
    expanded_tensor = reshaped_tensor.expand((4, 3))
    sum_tensor_1 = expanded_tensor.sum(0)
    sum_tensor_2 = sum_tensor_1.sum(0)

    sum_tensor_2.backward()

    assert all(tensor.grad == Buffer([4.0, 4.0, 4.0]))

    # Max reduction
    tensor = Tensor([3., 5., 7.], requires_grad=True)

    reshaped_tensor = tensor.reshape((1, 3))
    expanded_tensor = reshaped_tensor.expand((4, 3))
    sum_tensor_1 = expanded_tensor.max(0)
    sum_tensor_2 = sum_tensor_1.max(0)

    sum_tensor_2.backward()

    assert all(tensor.grad == Buffer([0.0, 0.0, 1.0]))


def test_broadcast_with_all_ops():
    ops = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y]

    # Scalars
    scalar_data = 3.0
    array_data = [-2.0, -1.0, 3.0, 4.0]
    x = Tensor(scalar_data)
    y = Tensor(array_data)
    for op in ops:
        expected_output_left_broadcasting = [op(scalar_data, e) for e in array_data]
        expected_output_right_broadcasting = [op(e, scalar_data) for e in array_data]

        # Left broadcasting
        result = op(x, y)
        assert all(result.buffer == Buffer(expected_output_left_broadcasting))

        # Right broadcasting
        result = op(y, x)
        assert all(result.buffer == Buffer(expected_output_right_broadcasting))

    # Multidimensional tensors
    contiguous_data_x = [3.0]
    contiguous_data_y = [-7.0, 9.0, 15.0, 17.0]
    x = Tensor(contiguous_data_x)
    y = Tensor([[contiguous_data_y]])
    for op in ops:
        expected_output_left_broadcasting = [[[op(contiguous_data_x[0], e) for e in contiguous_data_y]]]
        expected_output_right_broadcasting = [[[op(e, contiguous_data_x[0]) for e in contiguous_data_y]]]

        # Left broadcasting
        result = op(x, y)
        assert all(result.buffer == Buffer(expected_output_left_broadcasting))

        # Right broadcasting
        result = op(y, x)
        assert all(result.buffer == Buffer(expected_output_right_broadcasting))

    # Only need reshape
    contiguous_data_x = [3.0]
    contiguous_data_y = [[-7.0]]
    x = Tensor(contiguous_data_x)
    y = Tensor(contiguous_data_y)
    for op in ops:
        expected_output_left_broadcasting = [[op(contiguous_data_x[0], contiguous_data_y[0][0])]]
        expected_output_right_broadcasting = [[op(contiguous_data_y[0][0], contiguous_data_x[0])]]

        # Left broadcasting
        result = op(x, y)
        assert all(result.buffer == Buffer(expected_output_left_broadcasting))

        # Right broadcasting
        result = op(y, x)
        assert all(result.buffer == Buffer(expected_output_right_broadcasting))

    # Only need expand
    contiguous_data_x = [[3.0, 5.0, 13.0]]
    contiguous_data_y = [[-7.0, 9.0, 341.0], [14.0, -2.0, -342.0]]
    x = Tensor(contiguous_data_x)
    y = Tensor(contiguous_data_y)
    for op in ops:
        expected_output_left_broadcasting = [
            [op(e1, e2) for e1, e2 in zip(contiguous_data_x[0], contiguous_data_y[0])],
            [op(e1, e2) for e1, e2 in zip(contiguous_data_x[0], contiguous_data_y[1])]
        ]
        expected_output_right_broadcasting = [
            [op(e2, e1) for e1, e2 in zip(contiguous_data_x[0], contiguous_data_y[0])],
            [op(e2, e1) for e1, e2 in zip(contiguous_data_x[0], contiguous_data_y[1])]
        ]

        # Left broadcasting
        result = op(x, y)
        assert all(result.buffer == Buffer(expected_output_left_broadcasting))

        # Right broadcasting
        result = op(y, x)
        assert all(result.buffer == Buffer(expected_output_right_broadcasting))


def test_all_ops():
    # Test multiple operations
    tensor1 = Tensor([[1.2, 1.4]], requires_grad=True)
    tensor2 = Tensor([[2.0, -1.0]], requires_grad=True)

    assert tensor1.grad is None
    assert tensor2.grad is None

    result1 = tensor1.log() + tensor2.exp()
    result2 = tensor1 - tensor2
    result3 = result1 * result2
    result4 = result1 / result2
    result5 = (-result3) ** 2 + result1
    result6 = result5.relu()
    result7 = result6.maximum(result4)
    result8 = result7.sum((0, 1)) + result2.max((0, 1))

    assert result1.grad is None
    assert result1.grad_fn is not None
    assert result2.grad is None
    assert result2.grad_fn is not None
    assert result3.grad is None
    assert result3.grad_fn is not None
    assert result4.grad is None
    assert result4.grad_fn is not None
    assert result5.grad is None
    assert result5.grad_fn is not None
    assert result6.grad is None
    assert result6.grad_fn is not None
    assert result7.grad is None
    assert result7.grad_fn is not None
    assert result8.grad is None
    assert result8.grad_fn is not None

    result8.backward()

    assert all((tensor1.grad - Buffer([[-82.811744689941, 9.891427993774]])) < Buffer([[1e-05, 1e-05]]))
    assert all((tensor1.grad - Buffer([[-82.811744689941, 9.891427993774]])) > Buffer([[-1e-05, -1e-05]]))

    assert all((tensor2.grad - Buffer([[1.707202911377e+02, -2.843236923218e-02]])) < Buffer([[1e-05, 1e-05]]))
    assert all((tensor2.grad - Buffer([[1.707202911377e+02, -2.843236923218e-02]])) > Buffer([[-1e-05, -1e-05]]))

    assert result1.grad_fn is None
    assert result2.grad_fn is None
    assert result3.grad_fn is None
    assert result4.grad_fn is None
    assert result5.grad_fn is None
    assert result6.grad_fn is None
    assert result7.grad_fn is None
    assert result8.grad_fn is None


def test_softmax():
    # Wrong axis type
    with pytest.raises(TypeError):
        Tensor([1., 2.]).softmax(None)

    with pytest.raises(TypeError):
        Tensor([1., 2.]).softmax((0,))

    # Softmax for scalar values is not implemented
    with pytest.raises(ValueError):
        Tensor(1).softmax(0)

    # 1D
    tensor = Tensor([1., 0.2, 0.1])
    expected_output = Buffer([0.538822531700, 0.242108568549, 0.219068899751])

    softmax = tensor.softmax(axis=0)

    assert softmax.shape == (3,)
    assert all((softmax.buffer - expected_output) < Buffer([1e-05, 1e-05, 1e-05]))
    assert all((softmax.buffer - expected_output) > Buffer([-1e-05, -1e-05, -1e-05]))
    assert all(softmax.sum(0, keepdim=True).buffer == 1.0)

    # 2D
    tensor = Tensor([[1., 0.2, 0.1], [0.1, 1., 0.2]])
    expected_output_dim0 = Buffer(
        [[0.710949480534, 0.310025542974, 0.475020825863], [0.289050489664, 0.689974486828, 0.524979174137]]
    )
    expected_output_dim1 = Buffer(
        [[0.538822531700, 0.242108568549, 0.219068899751], [0.219068899751, 0.538822531700, 0.242108568549]]
    )

    softmax_dim0 = tensor.softmax(axis=0)
    softmax_dim1 = tensor.softmax(axis=1)

    assert softmax_dim0.shape == (2, 3)
    assert softmax_dim1.shape == (2, 3)

    positive_tolerance = Buffer([[1e-05, 1e-05, 1e-05], [1e-05, 1e-05, 1e-05]])
    negative_tolerance = Buffer([[-1e-05, -1e-05, -1e-05], [-1e-05, -1e-05, -1e-05]])

    assert all((softmax_dim0.buffer - expected_output_dim0) < positive_tolerance)
    assert all((softmax_dim0.buffer - expected_output_dim0) > negative_tolerance)
    assert all((softmax_dim1.buffer - expected_output_dim1) < positive_tolerance)
    assert all((softmax_dim1.buffer - expected_output_dim1) > negative_tolerance)

    assert all((softmax_dim0.sum(0, keepdim=True).buffer - 1.0) < 1e-08)
    assert all((softmax_dim0.sum(0, keepdim=True).buffer - 1.0) > -1e-08)

    assert all((softmax_dim1.sum(1, keepdim=True).buffer - 1.0) < 1e-08)
    assert all((softmax_dim1.sum(1, keepdim=True).buffer - 1.0) > -1e-08)

    # 3D
    tensor = Tensor([[[1., 0.2, 0.1], [0.1, 1., 0.2]]])
    expected_output_dim0 = Buffer(
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    expected_output_dim1 = Buffer(
        [[[0.710949480534, 0.310025542974, 0.475020825863], [0.289050489664, 0.689974486828, 0.524979174137]]]
    )
    expected_output_dim2 = Buffer(
        [[[0.538822531700, 0.242108568549, 0.219068899751], [0.219068899751, 0.538822531700, 0.242108568549]]]
    )

    softmax_dim0 = tensor.softmax(axis=0)
    softmax_dim1 = tensor.softmax(axis=1)
    softmax_dim2 = tensor.softmax(axis=2)

    assert softmax_dim0.shape == (1, 2, 3)
    assert softmax_dim1.shape == (1, 2, 3)

    positive_tolerance = Buffer([[[1e-05, 1e-05, 1e-05], [1e-05, 1e-05, 1e-05]]])
    negative_tolerance = Buffer([[[-1e-05, -1e-05, -1e-05], [-1e-05, -1e-05, -1e-05]]])

    assert all((softmax_dim0.buffer - expected_output_dim0) < positive_tolerance)
    assert all((softmax_dim0.buffer - expected_output_dim0) > negative_tolerance)
    assert all((softmax_dim1.buffer - expected_output_dim1) < positive_tolerance)
    assert all((softmax_dim1.buffer - expected_output_dim1) > negative_tolerance)
    assert all((softmax_dim2.buffer - expected_output_dim2) < positive_tolerance)
    assert all((softmax_dim2.buffer - expected_output_dim2) > negative_tolerance)

    assert all((softmax_dim0.sum(0, keepdim=True).buffer - 1.0) < 1e-08)
    assert all((softmax_dim0.sum(0, keepdim=True).buffer - 1.0) > -1e-08)

    assert all((softmax_dim1.sum(1, keepdim=True).buffer - 1.0) < 1e-08)
    assert all((softmax_dim1.sum(1, keepdim=True).buffer - 1.0) > -1e-08)

    assert all((softmax_dim2.sum(2, keepdim=True).buffer - 1.0) < 1e-08)
    assert all((softmax_dim2.sum(2, keepdim=True).buffer - 1.0) > -1e-08)


def test_gradient_function_backward_with_softmax():
    # Test backward propagation in GradientFunction with softmax
    for requires_grad in [True, False]:
        tensor = Tensor([[1., 0.2, 0.1], [0.1, 1., 0.2]], requires_grad=requires_grad)

        assert tensor.grad is None
        a = tensor.softmax(0)
        b = a.sum(0)
        result = b.sum(0)

        assert result.grad is None
        if requires_grad:
            assert result.grad_fn is not None
        else:
            assert result.grad_fn is None

        result.retain_grad()
        result.backward()

        if requires_grad:
            expected_grad = Buffer(
                [[2.118794562023e-08, -9.239481180146e-09, 0.000000000000e+00],
                 [8.614375879290e-09, -2.056284209573e-08, 0.000000000000e+00]]
            )

            assert all((tensor.grad - expected_grad) < 1e-05)
            assert all((tensor.grad - expected_grad) > -1e-05)
            assert all(result.grad == Buffer(1.0))

            assert result.grad_fn is None

        else:
            assert tensor.grad is None
            assert result.grad is None

            assert result.grad_fn is None


def test_init_methods():

    for requires_grad in [True, False]:
        # Empty tensor
        tensor = Tensor.uniform((), requires_grad=requires_grad)
        assert tensor.shape == ()
        assert tensor.dtype == DType.float32
        assert tensor.requires_grad == requires_grad

        # 1D
        tensor = Tensor.uniform((54,), requires_grad=requires_grad)
        assert tensor.shape == (54,)
        assert tensor.dtype == DType.float32
        assert all(tensor.buffer >= 0)
        assert all(tensor.buffer <= 1.0)
        assert tensor.requires_grad == requires_grad

        # 2D
        tensor = Tensor.uniform((16, 32), requires_grad=requires_grad)
        assert tensor.shape == (16, 32)
        assert tensor.dtype == DType.float32
        assert all(tensor.buffer >= 0)
        assert all(tensor.buffer <= 1.0)
        assert tensor.requires_grad == requires_grad

        # 3D
        tensor = Tensor.uniform((12, 13, 7), requires_grad=requires_grad)
        assert tensor.shape == (12, 13, 7)
        assert tensor.dtype == DType.float32
        assert all(tensor.buffer >= 0)
        assert all(tensor.buffer <= 1.0)
        assert tensor.requires_grad == requires_grad


def test_permute():
    # Multiple permute operations
    tensor = Tensor([[1., 2., 3.,], [4., 5., 6.]], requires_grad=True)

    new_tensor_1 = tensor.permute((1, 0))
    assert new_tensor_1.shape == (3, 2)

    new_tensor_2 = new_tensor_1.permute((1, 0))
    assert all(new_tensor_2.buffer == tensor.buffer)

    new_tensor_2.sum((0, 1)).backward()

    assert all(tensor.grad == Buffer([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))


def test_transpose():
    # Multiple transpose operations
    for requires_grad in [True, False]:
        tensor = Tensor([[[1., 2., 3.,], [4., 5., 6.]]], requires_grad=requires_grad)

        new_tensor_1 = tensor.transpose(2, 1)
        assert new_tensor_1.shape == (1, 3, 2)

        new_tensor_2 = new_tensor_1.transpose(1, 2)
        assert all(new_tensor_2.buffer == tensor.buffer)

        result = new_tensor_2.sum((0, 1, 2))

        assert result.grad is None
        if requires_grad:
            assert result.grad_fn is not None
        else:
            assert result.grad_fn is None

        result.retain_grad()
        result.backward()

        if requires_grad:
            assert all(tensor.grad == Buffer([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]))
            assert all(result.grad == Buffer(1.0))
        else:
            assert tensor.grad is None
            assert result.grad is None


def test_dot():
    # (2,) * (2,) -> ()
    tensor1 = Tensor([1.0, 2.0], requires_grad=True)
    tensor2 = Tensor([3.0, 4.0], requires_grad=True)
    assert all(tensor1.dot(tensor2).buffer == Buffer(11.0))

    # (2,2) * (2, 2) -> (2, 2)
    tensor1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    tensor2 = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    assert all(tensor1.dot(tensor2).buffer == Buffer([[19.0, 22.0], [43.0, 50.0]]))

    # (2,) * (2, 3) -> (3,)
    tensor1 = Tensor([1.0, 2.0], requires_grad=True)
    tensor2 = Tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], requires_grad=True)
    assert all(tensor1.dot(tensor2).buffer == Buffer([15.0, 18.0, 21.0]))

    # (4,2) * (2, 3) -> (4, 2)
    tensor1 = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    tensor2 = Tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], requires_grad=True)
    assert all(tensor1.dot(tensor2).buffer == Buffer(
            [[15.0, 18.0, 21.0], [33.0, 40.0, 47.0], [51.0, 62.0, 73.0], [69.0, 84.0, 99.0]]
        )
    )

    # (2, 4, 2) * (2, 2, 2) -> (2, 4, 2)
    tensor1 = Tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]],
        requires_grad=True
    )
    tensor2 = Tensor([[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]]], requires_grad=True)
    assert all(tensor1.dot(tensor2).buffer == Buffer(
            [
                [[55.0, 58.0], [127.0, 134.0], [199.0, 210.0], [271.0, 286.0]],
                [[419.0, 438.0], [507.0, 530.0], [595.0, 622.0], [683.0, 714.0]]
            ]
        )
    )

    # (3, 4, 2) * (2,) -> (3, 4)
    tensor1 = Tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[18.0, 19.0], [20.0, 21.0], [22.0, 23.0], [24.0, 25.0]]
        ],
        requires_grad=True
    )

    tensor2 = Tensor([1.0, 2.0], requires_grad=True)
    assert all(tensor1.dot(tensor2).buffer == Buffer(
            [[5.0, 11.0, 17.0, 23.0], [29.0, 35.0, 41.0, 47.0], [56.0, 62.0, 68.0, 74.0]]
        )
    )


def test_dot_backward():

    for require_grad in [True, False]:
        # (3, 4, 2) * (3, 2, 3) -> (3, 4, 3)
        tensor1 = Tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                [[18.0, 19.0], [20.0, 21.0], [22.0, 23.0], [24.0, 25.0]]
            ],
            requires_grad=require_grad
        )

        tensor2 = Tensor(
            [
                [[17.0, 18.0, 19.0], [20.0, 21.0, 22.0]],
                [[23.0, 24.0, 25.0], [26.0, 27.0, 28.0]],
                [[29.0, 30.0, 31.0], [32.0, 33.0, 34.0]]
            ],
            requires_grad=require_grad
        )

        result = tensor1.dot(tensor2).sum((0, 1, 2))
        result.backward()

        if require_grad:
            assert all(tensor1.grad == Buffer(
                    [
                        [[54.0, 63.0], [54.0, 63.0], [54.0, 63.0], [54.0, 63.0]],
                        [[72.0, 81.0], [72.0, 81.0], [72.0, 81.0], [72.0, 81.0]],
                        [[90.0, 99.0], [90.0, 99.0], [90.0, 99.0], [90.0, 99.0]]
                    ]
                )
            )

            assert all(tensor2.grad == Buffer(
                    [
                        [[16.0, 16.0, 16.0], [20.0, 20.0, 20.0]],
                        [[48.0, 48.0, 48.0], [52.0, 52.0, 52.0]],
                        [[84.0, 84.0, 84.0], [88.0, 88.0, 88.0]]
                    ]
                )
            )

        else:
            assert tensor1.grad is None
            assert tensor2.grad is None


def test_dead_branch():
    # Check gradient propagate even when there are dead branches on the DAG
    a = Tensor(1.0, requires_grad=True)
    b = Tensor(1.0, requires_grad=True)

    # Create the DAG with a dead branch
    # a, b  --->  c = a + b  --->  d = c * 2.0  --->  f = d / 4.0
    #                    \
    #                     \-->  e = c + 2.0  (Dead branch)
    c = a + b
    d = c * 2.0
    e = c + 2.0
    f = d / 4.0

    # Verify that `e` has been calculated
    assert all(e.buffer == Buffer(4.0))

    # Do the backward pass
    f.backward()

    # Check that a and b have received the expected gradients
    assert all(a.grad == Buffer(0.5))
    assert all(b.grad == Buffer(0.5))


def test_retain_graph():
    # Check retain_graph works
    a = Tensor(1.0, requires_grad=True)
    b = Tensor(1.0, requires_grad=True)

    # Create the DAG
    c = a + b
    d = c * 2.0
    f = d / 4.0

    # Do the backward pass
    f.backward(retain_graph=True)

    assert f.grad_fn is not None
    assert d.grad_fn is not None
    assert c.grad_fn is not None

    # Check that a and b have received the expected gradients
    assert all(a.grad == Buffer(0.5))
    assert all(b.grad == Buffer(0.5))

    # Do the backward pass a second time
    f.backward()

    assert f.grad_fn is None
    assert d.grad_fn is None
    assert c.grad_fn is None

    # Check that a and b have received the expected gradients
    assert all(a.grad == Buffer(1.0))
    assert all(b.grad == Buffer(1.0))

    # Now the graph has been released and we cannot do the backward pass again
    with pytest.raises(RuntimeError):
        f.backward(retain_graph=True)
    
    with pytest.raises(RuntimeError):
        f.backward(retain_graph=False)


def test_wrong_incoming_gradient():    
    # Check backward() method throws an error when incoming_gradient is not correct
    a = Tensor(1.0, requires_grad=True)
    b = Tensor(1.0, requires_grad=True)

    c = a + b
    d = a * b
    e = c / d

    # Call backward with the wrong value
    with pytest.raises(TypeError):
        e.backward(incoming_gradient=True)

    with pytest.raises(RuntimeError):
        e.backward(incoming_gradient=Buffer([1.0, 1.0]))

    with pytest.raises(TypeError):
        e.backward(incoming_gradient=Buffer(True))

    # Check with a valid incoming_gradient
    e.backward(incoming_gradient=Buffer(1.0))

    assert all(a.grad == Buffer(-1.0))
    assert all(b.grad == Buffer(-1.0))


def test_retain_grad():
    # Check we have access to the gradients on non-leaf tensors if we call retain_grad() on them
    a = Tensor(1.0, requires_grad=True)
    b = a * 3.0
    c = a / 2.0
    d = c * b

    d.backward()

    assert b.grad is None
    assert c.grad is None
    assert d.grad is None

    # Retain the gradients
    a = Tensor(1.0, requires_grad=True)
    b = a * 3.0
    c = a / 2.0
    d = c * b

    b.retain_grad()
    c.retain_grad()
    d.retain_grad()

    d.backward()

    assert all(b.grad  == Buffer(0.5))
    assert all(c.grad == Buffer(3.0))
    assert all(d.grad == Buffer(1.0))


def test_detach():
    # Try to detach tensors from the computational graph
    a = Tensor([2.0] * 20, requires_grad=True)

    # Detach the tensor
    a_detached = a.detach()
    assert not a_detached.requires_grad
    assert a is not a_detached
    assert a.buffer is a_detached.buffer

    # Modifications on the detached tensor doesn't affect the original one because it get copy after the operation
    a_detached += Tensor([1.0] * 20, requires_grad=False)
    assert a.buffer is not a_detached.buffer
    assert all(a.buffer != a_detached.buffer)

    # Do some operations with both versions (gradient doesn't flow back to a through c because we detached a)
    b = a ** 3
    c = a_detached ** 6
    i = (b + c).sum((0,))
    
    i.backward()

    assert all(a.grad == 12.0)
    assert a_detached.grad is None


def test_to_python():

    for dtype in DType:
        for value in [dtype.cast(-1), dtype.cast(0), dtype.cast(1)]:
            for data in [value, [value], [], [value, value, value], [[value, value], [value, value]], [[], []]]:
                # Create a Tensor
                tensor = Tensor(data)

                # Get the Python value
                python_data = tensor.to_python()

                # Check is the expected value
                assert isinstance(python_data, type(data))
                assert python_data == data