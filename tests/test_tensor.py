import pytest

from tinygpt.tensor import Tensor
from tinygpt.tensor import Buffer
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


def test_sum():
    # Test addition of two tensors
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 + tensor2
    assert result.shape == (3,)
    assert not result.requires_grad
    assert all(result.buffer == Buffer([5, 7, 9]))

    # Test addition of tensor and scalar
    tensor1 = Tensor([[[1.], [2.]], [[3.], [4.]]], requires_grad=True)
    tensor2 = Tensor([[[5.], [6.]], [[7.], [8.]]])
    result = tensor1 + tensor2
    assert result.shape == (2, 2, 1)
    assert result.requires_grad
    assert all(result.buffer == Buffer([[[6.], [8.]], [[10.], [12.]]]))


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


def test_gradient_function_backward_with_sum():
    # Test backward propagation in GradientFunction with sum
    for requires_grad_tensor1, requires_grad_tensor2 in [(True, True), (True, False), (False, True)]:
        tensor1 = Tensor(1.0, requires_grad=requires_grad_tensor1)
        tensor2 = Tensor(2.0, requires_grad=requires_grad_tensor2)

        assert tensor1.grad is None
        assert tensor2.grad is None

        result = tensor1 + tensor2

        assert result.grad is None
        assert result.grad_fn is not None

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
    # Test backward propagation in GradientFunction with sum
    for requires_grad_tensor1, requires_grad_tensor2 in [(True, True), (True, False), (False, True)]:
        tensor1 = Tensor(1.0, requires_grad=requires_grad_tensor1)
        tensor2 = Tensor(2.0, requires_grad=requires_grad_tensor2)

        assert tensor1.grad is None
        assert tensor2.grad is None

        result = tensor1 - tensor2

        assert result.grad is None
        assert result.grad_fn is not None

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

        result.backward()

        if requires_grad_tensor1:
            assert all(tensor1.grad == Buffer(1.0/3.0))
        else:
            assert tensor1.grad is None

        if requires_grad_tensor2:
            assert all(tensor2.grad == Buffer(-(2.0 / 3 ** 2)))
        else:
            assert tensor2.grad is None

        assert all(result.grad == Buffer(1.0))
        assert result.grad_fn is None


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
