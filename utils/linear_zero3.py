import os
from typing import Callable, List, Optional
import warnings
from torch import nn

from utils.globals import get_global_memory_buffer
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_model_parallel_group,
)
from torch.cuda.amp import custom_bwd, custom_fwd
from .comm import prepare_input_tensors_for_wgrad_compute


class LinearZeROFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        world_size = get_model_parallel_world_size()

        dim_size = list(weight.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_weight = get_global_memory_buffer().get_tensor(
            dim_size, input.dtype, "mpu"
        )
        torch.distributed.all_gather_into_tensor(
            all_gather_weight, weight, group=get_model_parallel_group()
        )
        output = torch.matmul(input, all_gather_weight.t())

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        group = get_model_parallel_group()

        dim_size = list(weight.size())
        world_size = get_model_parallel_world_size()
        dim_size[0] = dim_size[0] * world_size

        all_gather_weight = get_global_memory_buffer().get_tensor(
            dim_size, input.dtype, "mpu"
        )
        torch.distributed.all_gather_into_tensor(all_gather_weight, weight, group=group)

        total_input = input
        grad_input = grad_output.matmul(all_gather_weight)

        grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
            grad_output, total_input
        )

        grad_weight_global = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        grad_weight = torch.empty(
            weight.size(),
            dtype=input.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        torch.distributed.reduce_scatter_tensor(
            grad_weight, grad_weight_global, group=group
        )

        return grad_input, grad_weight, grad_bias, None, None, None, None


def linear_with_zero(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:

    args = [
        input,
        weight,
        bias,
    ]

    return LinearZeROFunc.apply(*args)


class LinearZeRO3(nn.Module):
    def __init__(self, input_features, output_features, bias=False):
        super(LinearZeRO3, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.world_size = get_model_parallel_world_size()
        self.group = get_model_parallel_group()

        self.weight = nn.Parameter(
            torch.Tensor(output_features // self.world_size, input_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features // self.world_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        self._forward_impl = linear_with_zero

        out = self._forward_impl(
            input=input,
            weight=self.weight,
            bias=self.bias,
        )
        return out
