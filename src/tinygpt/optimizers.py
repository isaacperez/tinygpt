import json
from typing import List, Union, Tuple

from tinygpt.tensor import Tensor 
from tinygpt.module import Module
from tinygpt.utils import tree_map, tree_flatten


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
        weights = file_or_weights
        if isinstance(weights, str):
            with open(weights, mode='r') as json_file:
                raw_weights = json.load(json_file)
            
            # We store the weights as str so we have to convert them back to Tensor
            def from_str_to_tensor(v):
                if isinstance(v, str) and Tensor.validate_serialized_tensor(v):
                    return Tensor.deserialize_tensor(v)  
                else:
                    return v

            weights = tree_map(from_str_to_tensor, raw_weights)

        if strict:
            new_weights = dict(weights)
            curr_weights = dict(tree_flatten(self.state))
            if extras := (new_weights.keys() - curr_weights.keys()):
                extras = " ".join(extras)
                raise ValueError(f"Received parameters not in the optimizer: {extras}.")

            if missing := (curr_weights.keys() - new_weights.keys()):
                missing = " ".join(missing)
                raise ValueError(f"Missing parameters: {missing}.")

            for k, v in curr_weights.items():
                v_new = new_weights[k]
                if not isinstance(v_new, Tensor):
                    raise ValueError(f"Expected dict but received {type(v_new)} for parameter {k}")
                if v_new.shape != v.shape:
                    raise ValueError(f"Expected shape {v.shape} but received shape {v_new.shape} for parameter {k}")

        self.state = weights
    
    def save_state(self, file: str) -> None:
        # We need a custom method to transform Tensor into a JSON format
        def serializer(obj):
            if isinstance(obj, Tensor):
                return obj.serialize_tensor()
            else:
                return obj.__dict__

        if file.endswith(".json"):
            with open(file, mode='w') as json_file:
                json.dump(self.state, json_file, indent=2, default=serializer)
        else:
            raise ValueError("Unsupported file extension. Use '.json'.")


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


class Adam(Optimizer):
    r"""The Adam optimizer [1].

    Our Adam implementation follows the original paper and omits the bias
    correction in the first and second moment estimates. In detail,

    [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
    optimization. ICLR 2015.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}}

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
    """

    def __init__(
        self,
        module: Module,
        learning_rate: float,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
    ) -> None:
        super().__init__(module=module)

        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps

    def apply_single(self, parameter: Tensor, state: OptimizerState) -> Tensor:
        """Performs the Adam parameter update and stores :math:`v` and :math:`m` in the optimizer state."""
        if parameter.grad is None:
            return parameter.detach()    

        b1, b2 = self.betas
        m = state. get("m", Tensor.zeros(parameter.grad.shape))
        v = state.get("v", Tensor.zeros(parameter.grad.shape))

        m = b1 * m + (1 - b1) * parameter.grad
        v = b2 * v + (1 - b2) * (parameter.grad ** 2)

        state["m"] = m
        state["v"] = v

        return parameter.detach() - self.learning_rate * m / ((v ** 0.5) + self.eps)
