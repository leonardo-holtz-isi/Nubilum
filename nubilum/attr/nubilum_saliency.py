#!/usr/bin/env python3

from typing import Any, Callable
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
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
                  abs: bool = False,
                  additional_forward_args: Any = None) -> TensorOrTupleOfTensorsGeneric:
        """
        Calls the attribute method from Captum Saliency.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensors for which
            saliency is computed.
            target (TargetType, optional): The target index in which the attributes
            will be computed . Defaults to None.
            abs (bool, optional): Returns absolute value of gradients if set to True,
            otherwise returns the (signed) gradients if False. Defaults to False.
            additional_forward_args (Any, optional): Additional arguments to pass to
            the forward function. Defaults to None.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attribution tensors.
        """
        return super().attribute(inputs, target, abs, additional_forward_args)
