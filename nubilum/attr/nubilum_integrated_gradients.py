#!/usr/bin/env python3

from typing import Any, Union, Callable
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import IntegratedGradients


class NubilumIntegratedGradients(IntegratedGradients):
    """
    Integrated Gradients implementation dedicated to point cloud
    """
    def __init__(self, forward_func: Callable[..., Any]) -> None:
        super().__init__(forward_func)

    def attribute(self,
                  inputs: TensorOrTupleOfTensorsGeneric,
                  baselines: TensorOrTupleOfTensorsGeneric = None,
                  target: TargetType = None,
                  additional_forward_args: Any = None,
                  n_steps: int = 50,
                  method: str = 'riemann_middle',
                  internal_batch_size: Union[None, int] = None,
                  return_convergence_delta: bool = False) -> TensorOrTupleOfTensorsGeneric:
        """
        Calls the attribute method from Captum Integrated Gradients.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensors for
            which integrated gradients is computed.
            baselines (TensorOrTupleOfTensorsGeneric, optional):
            Baseline tensors. Baseline tensors shapes must be equal to the
            input tensors. Defaults to None.
            target (TargetType, optional): The target index in which the
            attributes will be computed . Defaults to None.
            additional_forward_args (Any, optional): Additional arguments
            to pass to the forward function. Defaults to None.
            n_steps (int, optional): Number of steps for the integral
            approximation. Defaults to 50.
            method (str, optional): Method to be used for the integral
            approximation. Defaults to 'riemann_middle'.
            internal_batch_size (Union[None, int], optional): Size of the
            batch. Defaults to None.
            return_convergence_delta (bool, optional): Indicates whether
            to return convergence delta or not. Defaults to False.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attribution tensors.
        """
        return super().attribute(inputs, baselines, target,
                                 additional_forward_args,
                                 n_steps, method, internal_batch_size,
                                 return_convergence_delta)
