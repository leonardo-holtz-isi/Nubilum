#!/usr/bin/env python3

from typing import Any, Union, Callable, List
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor, stack
from captum.attr import IntegratedGradients

class NubilumIntegratedGradients(IntegratedGradients):
    """
    Saliency implementation dedicated to point cloud
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
        """
        return super().attribute(inputs, baselines, target, additional_forward_args, n_steps, method, internal_batch_size, return_convergence_delta)
    
    