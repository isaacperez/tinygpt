from __future__ import annotations
import json
import textwrap
from typing import Any, Optional, Tuple, Callable, Union, List

from tinygpt.tensor import Tensor
from tinygpt.utils import tree_flatten, tree_unflatten


class Module(dict):

    __call__: Callable

    def __init__(self):
        """Should be called by the subclasses of `Module`"""
        self._no_grad = set()
        self._training = True

    @property
    def training(self) -> bool:
        """Boolean indicating if the model is in training mode"""
        return self._training

    def _extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        children = tree_flatten(self.children(), is_leaf=self.is_module)
        value = f"{type(self).__name__}({self._extra_repr()}"
        for k, v in children:
            value += "\n"
            value += textwrap.indent(f"({k}): {repr(v)}", prefix="  ")
        if children:
            value += "\n"
        value += ")"

        return value

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"{type(self)!r} has no attribute {key!r}")

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def load_weights(
        self,
        file_or_weights: Union[str, List[Tuple[str, Tensor]]],
        strict: bool = True,
    ) -> None:
        """
        Update the model's weights from a `.json` or a list.

        Args:
            file_or_weights (str or list(tuple(str, Tensor))): The path to the weights `.json` file or a list of pairs 
                of parameter names and arrays.
            strict (bool, optional): If `True` then checks that the provided weights exactly match the parameters of 
                the model. Otherwise, only the weights actually contained in the model are loaded and shapes are not 
                checked. Default: `True`.
        """
        weights = file_or_weights
        if isinstance(weights, str):
            with open(weights, mode='r') as json_file:
                raw_weights = list(json.load(json_file).items())

            # We store the weights as dict so we have to convert them back to Tensor
            weights = tree_flatten(raw_weights)
            for idx, (k, v) in enumerate(weights):
                if isinstance(v, str) and Tensor.validate_serialized_tensor(v):
                    weights[idx] = (k, Tensor.deserialize_tensor(v))
            weights = tree_unflatten(weights)

        if strict:
            new_weights = dict(weights)
            curr_weights = dict(tree_flatten(self.parameters()))
            if extras := (new_weights.keys() - curr_weights.keys()):
                extras = " ".join(extras)
                raise ValueError(f"Received parameters not in model: {extras}.")

            if missing := (curr_weights.keys() - new_weights.keys()):
                missing = " ".join(missing)
                raise ValueError(f"Missing parameters: {missing}.")

            for k, v in curr_weights.items():
                v_new = new_weights[k]
                if not isinstance(v_new, Tensor):
                    raise ValueError(f"Expected dict but received {type(v_new)} for parameter {k}")
                if v_new.shape != v.shape:
                    raise ValueError(f"Expected shape {v.shape} but received shape {v_new.shape} for parameter {k}")

        self.update(tree_unflatten(weights))

    def save_weights(self, file: str) -> None:
        """
        Save the model's weights to a JSON file.
        """
        params_dict = dict(tree_flatten(self.parameters()))

        # We need a custom method to transform Tensor into a JSON format
        def serializer(obj):
            if isinstance(obj, Tensor):
                return obj.serialize_tensor()
            else:
                return obj.__dict__

        if file.endswith(".json"):
            with open(file, mode='w') as json_file:
                json.dump(params_dict, json_file, indent=2, default=serializer)
        else:
            raise ValueError("Unsupported file extension. Use '.json'.")

    @staticmethod
    def is_module(value: Any) -> bool:
        return isinstance(value, Module)

    @staticmethod
    def valid_child_filter(module: Any, key: Any, value: Any) -> bool:
        return isinstance(value, (dict, list))

    @staticmethod
    def valid_parameter_filter(module: Any, key: Any, value: Any) -> bool:
        return isinstance(value, (dict, list, Tensor)) and not key.startswith("_")

    @staticmethod
    def trainable_parameter_filter(module: Any, key: Any, value: Any) -> bool:
        return Module.valid_parameter_filter(module, key, value) and key not in module._no_grad

    def filter_and_map(
        self,
        filter_fn: Callable[[Module, str, Any], bool],
        map_fn: Optional[Callable] = None,
        is_leaf_fn: Optional[Callable[[Module, str, Any], bool]] = None,
    ) -> dict:
        """Recursively filter the contents of the module using `filter_fn`, namely only select keys and values where 
        `filter_fn` returns true.

        This is used to implement :meth:`parameters` and :meth:`trainable_parameters` but it can also be used to extract 
        any subset of the module's parameters.

        Args:
            filter_fn (Callable): Given a value, the key in which it is found and the containing module, decide whether 
                to keep the value or drop it.
            map_fn (Callable, optional): Optionally transform the value before returning it.
            is_leaf_fn (Callable, optional): Given a value, the key in which it is found and the containing module 
                decide if it is a leaf.

        Returns:
            A dictionary containing the contents of the module recursively filtered
        """

        map_fn = map_fn or (lambda x: x)
        is_leaf_fn = is_leaf_fn or (
            lambda m, k, v: not isinstance(v, (Module, dict, list))
        )

        def unwrap(vk, v):
            if is_leaf_fn(self, vk, v):
                return map_fn(v)

            if isinstance(v, Module):
                return v.filter_and_map(filter_fn, map_fn, is_leaf_fn)

            if isinstance(v, dict):
                nd = {}
                for k, v in v.items():
                    tk = f"{vk}.{k}"
                    nd[k] = unwrap(tk, v) if filter_fn(self, tk, v) else {}
                return nd

            if isinstance(v, list):
                nl = []
                for i, vi in enumerate(v):
                    tk = f"{vk}.{i}"
                    nl.append(unwrap(tk, vi) if filter_fn(self, tk, vi) else {})
                return nl

            raise RuntimeError(f"Unexpected leaf found while traversing the module: {vk} - {type(v)}")

        return {k: unwrap(k, v) for k, v in self.items() if filter_fn(self, k, v)}

    def parameters(self) -> dict:
        """Recursively return all the :class:`Tensor` members of this Module as a dict of dicts and lists."""
        return self.filter_and_map(self.valid_parameter_filter)

    def trainable_parameters(self) -> dict:
        """Recursively return all the non frozen :class:`Tensor` members of this Module as a dict of dicts and lists."""
        return self.filter_and_map(self.trainable_parameter_filter)

    def children(self) -> dict:
        """Return the direct descendants of this Module instance."""
        return self.filter_and_map(self.valid_child_filter, is_leaf_fn=lambda m, k, v: isinstance(v, Module))

    def leaf_modules(self) -> dict:
        """Return the submodules that do not contain other modules."""

        def _is_leaf_module(m, k, v):
            return isinstance(v, Module) and len(tree_flatten(v.children())) == 0

        return self.filter_and_map(self.valid_child_filter, is_leaf_fn=_is_leaf_module)

    def update(self, parameters: dict) -> None:
        """Replace the parameters of this Module with the provided ones in the dict of dicts and lists.

        Commonly used by the optimizer to change the model to the updated (optimized) parameters. Also used by the 
        :meth:`tinygpt.nn.value_and_grad` to set the tracers in the model in order to compute gradients.

        The passed in parameters dictionary need not be a full dictionary similar to :meth:`parameters`. Only the 
        provided locations will be updated.

        Args:
            parameters (dict): A complete or partial dictionary of the modules parameters.
        """

        def apply(dst, parameters):
            if isinstance(parameters, dict):
                for k in parameters:
                    if k in dst:
                        current_value = dst[k]
                        new_value = parameters[k]
                        if isinstance(current_value, Tensor):
                            dst[k] = new_value
                        elif isinstance(current_value, Module):
                            current_value.update(new_value)
                        elif isinstance(current_value, (dict, list)):
                            apply(current_value, new_value)
            elif isinstance(parameters, list):
                for i in range(len(dst)):
                    current_value = dst[i]
                    new_value = parameters[i]
                    if isinstance(current_value, Tensor):
                        dst[i] = new_value
                    elif isinstance(current_value, Module):
                        current_value.update(new_value)
                    elif isinstance(current_value, (dict, list)):
                        apply(current_value, new_value)

        apply(self, parameters)

    def apply(
        self,
        map_fn: Callable[[Tensor], Tensor],
        filter_fn: Optional[Callable[[Module, str, Any], bool]] = None,
    ) -> None:
        """Map all the parameters using the provided `map_fn` and immediately update the module with the mapped 
        parameters.

        For instance running `model.apply(lambda x: x.relu())` apply the ReLU activation function to all parameters.

        Args:
            map_fn (Callable): Maps an array to another array
            filter_fn (Callable, optional): Filter to select which arrays to
                map (default: :meth:`Module.valid_parameter_filter`).
        """
        filter_fn = filter_fn or Module.valid_parameter_filter
        self.update(self.filter_and_map(filter_fn, map_fn))

    def update_modules(self, modules: dict) -> None:
        """Replace the child modules of this :class:`Module` instance with the provided ones in the dict of dicts and
        lists.

        It is the equivalent of :meth:`Module.update` but for modules instead of parameters and allows us to flexibly 
        edit complex architectures by programmatically swapping layers.

        The passed in parameters dictionary need not be a full dictionary similar to :meth:`parameters`. Only the 
        provided locations will be updated.

        Args:
            modules (dict): A complete or partial dictionary of the modules
                submodules.
        """

        def apply(dst, modules):
            if isinstance(modules, dict):
                for k in modules:
                    if k in dst:
                        current_value = dst[k]
                        new_value = modules[k]
                        if self.is_module(current_value) and self.is_module(new_value):
                            dst[k] = new_value
                        elif isinstance(current_value, (dict, list)):
                            apply(current_value, new_value)
            elif isinstance(modules, list):
                for i in range(len(dst)):
                    current_value = dst[i]
                    new_value = modules[i]
                    if self.is_module(current_value) and self.is_module(new_value):
                        dst[i] = new_value
                    elif isinstance(current_value, (dict, list)):
                        apply(current_value, new_value)

        apply(self, modules)

    def apply_to_modules(self, apply_fn: Callable[[str, Module], Any]) -> None:
        """Apply a function to all the modules in this instance (including this instance).

        Args:
            apply_fn (Callable): The function to apply to the modules.
        """
        module_stack = [("", self)]
        while module_stack:
            prefix, mod = module_stack.pop()
            apply_fn(prefix, mod)
            prefix = "." + prefix if prefix else ""
            module_stack.extend(tree_flatten(mod.children(), prefix=prefix, is_leaf=self.is_module))

    def modules(self) -> List[Module]:
        """Return a list with all the modules in this instance.

        Returns:
            A list of :class:`tinygpt.nn.Module` instances.
        """
        modulelist = []
        self.apply_to_modules(lambda k, m: modulelist.append(m))
        return modulelist

    def named_modules(self) -> List[(str, Module)]:
        """Return a list with all the modules in this instance and their name with dot notation.

        Returns:
            A list of tuples (str, :class:`tinygpt.nn.Module`).
        """
        modulelist = []
        self.apply_to_modules(lambda k, m: modulelist.append((k, m)))
        return modulelist

    def _validate_keys(self, keys, strict):
        keys = keys if isinstance(keys, list) else [keys]
        if strict:
            for k in keys:
                if k not in self:
                    raise KeyError(f"Module doesn't contain member {k}.")
        return keys

    def freeze(
        self,
        *,
        recurse: bool = True,
        keys: Optional[Union[str, List[str]]] = None,
        strict: bool = False,
    ) -> None:
        """Freeze the Module's parameters or some of them. Freezing a parameter means not computing gradients for it.

        This function is idempotent i.e. freezing a frozen model is a no-op.

        Example:
            For instance to only train the attention parameters from a Transformer:

            .. code-block:: python

                model = nn.Transformer()
                model.freeze()
                model.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith("attention") else None)

        Args:
            recurse (bool, optional): If True then freeze the parameters of the submodules as well. Default: `True`.
            keys (str or list[str], optional): If provided then only these parameters will be frozen otherwise all the 
                parameters of a module. For instance freeze all biases by calling `module.freeze(keys="bias")`.
            strict (bool, optional): If set to `True` validate that the passed keys exist. Default: `False`.
        """

        def _freeze_impl(_, m):
            local_keys = keys
            if local_keys is None:
                local_keys = tree_flatten(
                    m.filter_and_map(lambda m, k, v: (not isinstance(v, Module)) and m.valid_parameter_filter(m, k, v))
                )
                local_keys = [k for (k, v) in local_keys]

            local_keys = m._validate_keys(local_keys, strict)
            m._no_grad.update(local_keys)

        if recurse:
            self.apply_to_modules(_freeze_impl)
        else:
            _freeze_impl("", self)

    def unfreeze(
        self,
        *,
        recurse: bool = True,
        keys: Optional[Union[str, List[str]]] = None,
        strict: bool = False,
    ) -> None:
        """Unfreeze the Module's parameters or some of them.

        This function is idempotent ie unfreezing a model that is not frozen is a noop.

        Example:

            For instance to only train the biases of a Transformer one can do:

            .. code-block:: python

                model = nn.Transformer()
                model.freeze()
                model.unfreeze(keys="bias")

        Args:
            recurse (bool, optional): If True then unfreeze the parameters of the submodules as well. Default: `True`.
            keys (str or list[str], optional): If provided then only these parameters will be unfrozen otherwise all the
                parameters of a module. For instance unfreeze all biases by calling `module.unfreeze(keys="bias")`.
            strict (bool, optional): If set to `True` validate that the passed keys exist. Default: `False`.
        """

        def _unfreeze_impl(_, m):
            if keys is None:
                m._no_grad.clear()
            else:
                local_keys = m._validate_keys(keys, strict)
                m._no_grad.difference_update(local_keys)

        if recurse:
            self.apply_to_modules(_unfreeze_impl)
        else:
            _unfreeze_impl("", self)

    def train(self, mode: bool = True) -> None:
        """Set the model in or out of training mode.

        Training mode only applies to certain layers. For example
        :obj:`Dropout` applies a random mask in training mode, but is the
        identity in evaluation mode.

        Args:
            mode (bool): Indicate if the model should be in training or
                evaluation mode. Default: `True`.
        """

        def _set_train(_, m):
            m._training = mode

        self.apply_to_modules(_set_train)

    def eval(self) -> None:
        """Set the model to evaluation mode.

        See :func:`train`.
        """
        self.train(False)

    def zero_grad(self) -> None:
        # Zeros the gradient of all parameters in this module and its submodules
        self.apply(map_fn=lambda x: x.zero_grad(), filter_fn=self.trainable_parameter_filter)


class FullyConnectedLayer(Module):
    # A fully connected layer (or dense layer) which applies a linear transformation to the incoming data: `y = xW + b`

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()

        self.weights = Tensor.uniform(shape=(input_dims, output_dims), requires_grad=True)
        if bias:
            self.bias = Tensor.uniform(shape=(output_dims,), requires_grad=True)

    def _extra_repr(self) -> str:
        return f"input_dims={self.weights.shape[0]}, output_dims={self.weights.shape[1]}, bias={'bias' in self}"

    def __call__(self, x: Tensor) -> Tensor:
        if "bias" in self:
            return x.dot(self.weights) + self.bias
        else:
            return x.dot(self.weights)


class MLP(Module):

    activation_functions = {
        'linear': lambda tensor: tensor,
        'relu': lambda tensor: tensor.relu()
    }

    def __init__(self, input_dims: int,  hidden_dims: list[int], activation_fn: str, bias: bool = True) -> None:
        super().__init__()

        if activation_fn not in self.activation_functions:
            raise ValueError(
                f"Unknown activation function '{activation_fn}'. "
                f"Expecting one of {list(self.activation_functions.keys())}"
            )

        self.activation_fn = activation_fn
        self.layers = []
        for out_features in hidden_dims:
            self.layers.append(FullyConnectedLayer(input_dims, out_features, bias))
            input_dims = out_features

    def _extra_repr(self) -> str:
        return f"activation_fn='{self.activation_fn}'"

    def __call__(self, x: Tensor) -> Tensor:
        # Apply a sequence of linear transformations to the incoming tensor interleaving the activation function
        output = x
        for layer in self.layers:
            output = MLP.activation_functions[self.activation_fn](layer(output))

        return output