from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_model_parallel_group,
)
from fairscale.nn.model_parallel.mappings import (
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from fairscale.nn.model_parallel.utils import (
    VocabUtility,
    divide_and_check_no_remainder,
)

from fairscale.nn.model_parallel.layers import _initialize_affine_weight

from typing import Any


def _reducescatter(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    process_group_size = torch.distributed.get_world_size(group=group)
    if process_group_size == 1:
        return input_

    # ReduceScatter
    assert input_.dim() == 3, "Only 3D input is supported in _reducescatter"
    input_list_ = list(input_.chunk(process_group_size, dim=1))
    output_ = torch.empty_like(input_list_[0])
    torch.distributed.reduce_scatter(output_, input_list_, group=group)
    ctx.save_for_backward(output_)

    return output_


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    group = get_model_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    bs_, seqlen, _ = input_.shape
    assert bs_ == 1, f"Batch size should be 1, got {bs_}"
    input_ = input_.view(seqlen, -1)

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
    torch.distributed.reduce_scatter_tensor(output, input_.contiguous(), group=group)

    output = output.unsqueeze_(0)
    assert output.dim() == 3
    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""
    group = get_model_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    bs_, seqlen, _ = input_.shape
    assert bs_ == 1, f"Batch size should be 1, got {bs_}"
    input_ = input_.view(seqlen, -1)

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    output = output.unsqueeze_(0)
    assert output.dim() == 3
    return output


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


def reducescatter_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


class RowParallelLinearRS(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super(RowParallelLinearRS, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(
            in_features, world_size
        )

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(
            torch.Tensor(self.out_features, self.input_size_per_partition)
        )
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            1,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reducescatter_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
