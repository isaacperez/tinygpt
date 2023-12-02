import pytest

from tinygpt.tensor import Tensor
from tinygpt.utils import DType


def test_Tensor():
    # Scalars
    tensor = Tensor([])
    assert tensor.shape == (0,)
    assert tensor.ndim == 1
    assert tensor.dtype == DType.float32
    assert not tensor.requires_grad

    for scalar in [1.0, -1.0, 0.0, -1, 0, 1, True, False]:
        tensor = Tensor(scalar)
        assert tensor.shape == ()
        assert tensor.ndim == 0
        assert tensor.dtype == DType.deduce_dtype(scalar)
        assert not tensor.requires_grad

    # Lists
    tensor = Tensor([1, 2, 3])
    assert tensor.shape == (3,)
    assert tensor.ndim == 1
    assert tensor.dtype == DType.int32
    assert not tensor.requires_grad

    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    assert tensor.shape == (2, 3)
    assert tensor.ndim == 2
    assert tensor.dtype == DType.int32
    assert not tensor.requires_grad

    tensor = Tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert tensor.shape == (3, 2, 2)
    assert tensor.ndim == 3
    assert tensor.dtype == DType.int32
    assert not tensor.requires_grad

    # requires_grad
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

    # Try not valid dtypes to require gradients
    for data in ([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]], 1, True, False, [1, 2], [True, False]):
        with pytest.raises(
            RuntimeError,
            match="Only Tensors of floating point dtype can require gradients"
        ):
            tensor = Tensor(data, requires_grad=True)
