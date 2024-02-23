from typing import List, Union, Tuple

from tinygpt.utils import tree_map
from tinygpt.tensor import Tensor 
from tinygpt.nn import Module


class OptimizerState(dict):
    """The optimizer state implements a recursively defined :class:`collections.defaultdict`, namely a missing key in an 
    optimizer state is an :class:`OptimizerState`.

    .. note::
       :meth:`OptimizerState.get` in contrast to a normal dictionary also sets the key to the ``default`` value if the 
       ``key`` was not present in the dictionary.
    """

    def __getitem__(self, key):
        if key not in self:
            self[key] = OptimizerState()
        return super().__getitem__(key)

    def get(self, key, default):
        """If ``key`` doesn't exist set its value to ``default`` and then return it."""
        if key not in self:
            self[key] = default
        return super().__getitem__(key)


class Optimizer:
    """The base class for all optimizers. It allows us to implement an optimizer on a per-parameter basis and apply it 
    to a parameter tree.

    Attributes:
        state (OptimizerState): It holds the optimizer's state dictionary.
    """

    def __init__(self, module: Module) -> None:
        self.state = OptimizerState()
        self.module = module

    def update(self) -> None:
        """Apply the gradients to the parameters of the model and update the model with the new parameters."""
        update_tensor_with_new_one = lambda param, state: param.assign(self.apply_single(param, state))
        self.module.update(tree_map(update_tensor_with_new_one, self.module.trainable_parameters(), self.state))

    def apply_single(self, parameter: Tensor, state: OptimizerState) -> Tensor:
        """To be extended by the children classes to implement each optimizer's update."""
        raise NotImplementedError()

    def zero_grad(self) -> None:
        self.module.zero_grad()

    def load_state(self, file_or_weights: Union[str, List[Tuple[str, Tensor]]], strict: bool = True) -> None:
        raise NotImplementedError()
    
    def save_state(self, file: str) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    r"""The stochastic gradient descent optimizer.

    Updates a parameter :math:`w` with a gradient :math:`g` as follows

    .. math::

        v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
        w_{t+1} &= w_t - \lambda v_{t+1}

    Args:
        learning_rate (float): The learning rate :math:`\lambda`.
        momentum (float, optional): The momentum strength :math:`\mu`. Default: ``0``
        weight_decay (float, optional): The weight decay (L2 penalty). Default: ``0``
        dampening (float, optional): Dampening for momentum :math:`\tau`. Default: ``0``
        nesterov (bool, optional): Enables Nesterov momentum. Default: ``False``
    """

    def __init__(
        self,
        module: Module,
        learning_rate: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening."
            )
        super().__init__(module=module)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def apply_single(self, parameter: Tensor, state: OptimizerState) -> Tensor:
        """Performs the SGD parameter update and stores :math:`v` in the optimizer state."""
        if parameter.grad is None:
            return parameter.detach()

        if self.momentum <= 0:
            return parameter.detach() - self.learning_rate * parameter.grad

        v = state.get("v", Tensor.zeros(parameter.grad.shape))

        if self.weight_decay != 0:
            parameter.grad += self.weight_decay * parameter.detach()

        v = self.momentum * v
        if self.dampening > 0:
            v += (1 - self.dampening) * parameter.grad
        else:
            v += parameter.grad

        if self.nesterov:
            update = parameter.grad + self.momentum * v
        else:
            update = v

        state["v"] = v

        return parameter.detach() - self.learning_rate * update
