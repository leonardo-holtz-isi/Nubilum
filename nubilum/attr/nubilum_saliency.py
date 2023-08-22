#!/usr/bin/env python3

from typing import Any, Callable, List
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor, stack
from captum.attr import Saliency

class NubilumSaliency(Saliency):
    """
    Saliency implementation dedicated to point cloud
    """
    def __init__(self, forward_func: Callable[..., Any]) -> None:
        super().__init__(forward_func)
    
    def attribute(self,
                  inputs: TensorOrTupleOfTensorsGeneric,
                  target: TargetType = None,
                  abs: bool = True,
                  additional_forward_args: Any = None
        ) -> TensorOrTupleOfTensorsGeneric:
        """
        Calls the attribute method from Captum Saliency.
        """
        return super().attribute(inputs, target, abs, additional_forward_args)
        