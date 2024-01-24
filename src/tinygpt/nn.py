from __future__ import annotations
from collections import OrderedDict
from typing import Any, Iterator, Optional, Set, Tuple

from tinygpt.tensor import Tensor


class Module:

    def __init__(self) -> None:
        # super().__init__() should be called by your subclasses before they use any other methods defined in it
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

    def zero_grad(self) -> None:
        # Zeros the gradient of all parameters in this module and its submodules
        for p in self.parameters():
            p.zero_grad()

    def _save_to_state_dict(self, destination: Optional[dict] = None, prefix: str = '') -> None:
        # Save the module's state to the provided dictionary. Used internally for state_dict
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param

    def state_dict(self, destination: Optional[dict] = None, prefix: str = '') -> dict:
        # Returns a dictionary containing references to the whole state of the module
        if destination is None:
            destination = OrderedDict()

        self._save_to_state_dict(destination, prefix)

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.')

        return destination

    def _load_from_state_dict(self, state_dict: dict, prefix: str) -> None:
        # Load the module's state from the given state dictionary. Used internally for load_state_dict.
        unexpected_keys_from_state_dict = list(state_dict.keys())
        for name, param in self._parameters.items():
            if param is None:
                continue

            key = prefix + name
            if key in state_dict:
                unexpected_keys_from_state_dict.remove(key)
                input_param = state_dict[key]

                # Check it's a tensor
                if not isinstance(input_param, Tensor):
                    raise TypeError(
                        f'Expecting type tensor for {key}, found type {type(input_param)} from checkpoint.'
                    )

                # Local shape should match the one in checkpoint
                if input_param.shape != param.shape:
                    raise RuntimeError(
                        f'size mismatch for {key}: copying a tensor with shape {input_param.shape} from checkpoint, '
                        f'the shape in current model is {param.shape}.'
                    )

                # Add the attribute to the module
                try:
                    setattr(self, name, input_param)
                except Exception as ex:
                    raise RuntimeError(
                        f'While copying the tensor named "{key}", '
                        f'whose shape in the model are {param.shape} and '
                        f'whose shape in the checkpoint are {input_param.shape}, '
                        f'an exception occurred : {ex.args}.'
                    )

            else:
                raise KeyError(f"Missing key {key} in state_dict")

        # Raise an error if there are unexpected keys in the state_dict
        if len(unexpected_keys_from_state_dict) > 0:
            raise KeyError(f'Found unexpected keys in the state_dict: {unexpected_keys_from_state_dict}')

    def load_state_dict(self, state_dict: dict) -> None:
        #  Copies parameters from `state_dict` into this module and its descendants
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected state_dict to be dict, got {type(state_dict)}.")

        def load(module, local_state_dict, prefix=''):
            module._load_from_state_dict(local_state_dict, prefix)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)

    def __setattr__(self, name: str, value: Any) -> None:
        # Set an attribute of the module. Custom logic is used to handle parameters and modules.

        # Ensure that super().__init__() has been called
        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign attribute before Module.__init__() call")

        # Validate the name
        if not isinstance(name, str):
            raise TypeError(f"attribute name should be a string. Got {type(name)}")
        elif '.' in name:
            raise KeyError("attribute name can't contain \".\"")
        elif name == '':
            raise KeyError("attribute name can't be empty string \"\"")
        elif name in ['_parameters', '_modules']:
            raise ValueError(f"name '{name}' is reserved")

        # Handle assignment of parameters and modules
        if name in self._parameters or isinstance(value, Tensor):
            if not isinstance(value, Tensor) and value is not None:
                raise TypeError(f"Cannot assign '{type(value)}' as attribute '{name}' (Tensor or None expected)")

            # Save it as parameter
            self._parameters[name] = value

        elif name in self._modules or isinstance(value, Module):
            if not isinstance(value, Module) and value is not None:
                raise TypeError(f"Cannot assign '{type(value)}' as attribute '{name}' (Module or None expected)")

            # Save it as module
            self._modules[name] = value

        else:
            # Regular attribute assignment
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        # Retrieve an attribute of the module, with custom handling for parameters and modules
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]

        if '_modules' in self.__dict__ and name in self._modules:
            return self._modules[name]

        if name in self.__dict__:
            return self.__dict__[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name):
        # Deletes an attribute from the module. Special handling for parameters and modules
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def modules(self) -> Iterator[Module]:
        # Returns an iterator over all modules in the network, including itself
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set['Module']] = None,
        prefix: str = '',
        remove_duplicate: bool = True
    ) -> Iterator[Module]:
        # Returns an iterator over all modules in the network, yielding both the name and the module itself
        if memo is None:
            memo = set()

        if self not in memo:
            if remove_duplicate:
                memo.add(self)

            yield prefix, self

            for name, module in self._modules.items():
                if module is None:
                    continue

                submodule_prefix = prefix + ('.' if prefix else '') + name

                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    def children(self) -> Iterator['Module']:
        # Returns an iterator over immediate children modules of this module
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        # Return an iterator over immediate children modules, yielding both the name of the module and the module itself
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)

                yield name, module

    def parameters(self) -> Iterator[Tensor]:
        # Returns an iterator over all parameters of the module and its descendants
        for module in self.modules():
            for _, param in module.named_parameters():
                yield param

    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        # Return an iterator over all parameters of the module, yielding both the name and the parameter itself
        memo = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            for k, v in module._parameters.items():
                if v is None or v in memo:
                    continue

                if remove_duplicate:
                    memo.add(v)

                name = module_prefix + ('.' if module_prefix else '') + k

                yield name, v


class FullyConnectedLayer(Module):
    # A fully connected layer (or dense layer) which applies a linear transformation to the incoming data: `y = xW + b`

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True) -> None:
        # Initializes the FullyConnectedLayer with given dimensions and bias settings
        super().__init__()

        # Save the configuration
        self.in_features, self.out_features, self.use_bias = in_features, out_features, use_bias

        # Validate the type and values of the current configuration
        self.validate_configuration()

        # Create the weights of the layer
        self.weights = Tensor.uniform((self.in_features, self.out_features), requires_grad=True)
        if self.use_bias:
            self.bias = Tensor.uniform((self.out_features,), requires_grad=True)
        else:
            self.bias = None

    def validate_configuration(self) -> None:
        # Validate the types
        if not isinstance(self.in_features, int):
            raise TypeError(f"Expected in_features to be a int, found {type(self.in_features)}")
        
        if not isinstance(self.out_features, int):
            raise TypeError(f"Expected out_features to be a int, found {type(self.in_features)}")

        if not isinstance(self.use_bias, bool):
            raise TypeError(f"Expected activation to be a str, found {type(self.use_bias)}")
        
        # Validate the values
        if self.in_features < 1:
            raise ValueError(f"in_features has to be a positive number, found {self.in_features}")

        if self.out_features < 1:
            raise ValueError(f"out_features has to be a positive number, found {self.out_features}")

    def __call__(self, x: Tensor) -> Tensor:
        # Applies a linear transformation to the incoming tensor
        if self.use_bias:
            return x.dot(self.weights) + self.bias
        else:
            return x.dot(self.weights)

    def __repr__(self) -> str:
        return (
            f"FullyConnectedLayer (in_features={self.in_features}, out_features={self.out_features}, "
            f"use_bias={self.use_bias})")


class MLP(Module):

    activation_functions = {
        'linear': lambda tensor: tensor,
        'relu': lambda tensor: tensor.relu()
    }

    def __init__(self, in_features: list[int],  hidden_features: int, activation: str, use_bias: bool = True) -> None:
        super().__init__()
        
        # Save the configuration
        self.in_features = in_features 
        self.hidden_features = hidden_features
        self.activation = activation
        self.use_bias = use_bias

        # Validate the type and values of the current configuration
        self.validate_configuration()

        # Create the MLP with the expected pipeline
        self.activation_function = MLP.activation_functions[self.activation]
        self.layers = []
        for out_features in self.hidden_features:
            self.layers.append(FullyConnectedLayer(in_features, out_features, self.use_bias))
            in_features = out_features

    def validate_configuration(self) -> None:

        # Validate the types
        if not isinstance(self.in_features, int):
            raise TypeError(f"Expected in_features to be a int, found {type(self.in_features)}")
        
        if not isinstance(self.hidden_features, list):
            raise TypeError(f"Expected hidden_features to be a list, found {type(self.hidden_features)}")
        
        if not all(isinstance(val, int) for val in self.hidden_features):
            raise TypeError(
                f"Expected hidden_features elements to be a int, found {[type(val) for val in self.hidden_features]}"
            )
        
        if not isinstance(self.activation, str):
            raise TypeError(f"Expected activation to be a str, found {type(self.activation)}")
        
        if not isinstance(self.use_bias, bool):
            raise TypeError(f"Expected activation to be a str, found {type(self.activation)}")
        
        # Validate the values
        if self.in_features < 1:
            raise ValueError(f"in_features has to be a positive number, found {self.in_features}")
        
        if len(self.hidden_features) == 0:
            raise ValueError("hidden_feaures cannot be empty")
        
        if not all(val for val in self.hidden_features):
            raise ValueError(f"all values in hidden_feaures have to be a positive number, found {self.in_features}")
        
        if self.activation not in MLP.activation_functions:
            raise ValueError(
                f"activation function has to be one of {list(MLP.activation_functions.keys())}, found {self.activation}"
            )

    def __call__(self, x: Tensor) -> Tensor:
        # Apply a sequence of linear transformations to the incoming tensor interleaving the activation function
        output = x
        for layer in self.layers:
            output = self.activation_function(layer(output))

        return output

    def __repr__(self) -> str:
        return (
            f"MLP (in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"activation='{self.activation}', use_bias={self.use_bias})")
