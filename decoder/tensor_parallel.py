from decoder.odysseus import LlamaFlashAttention2TPSP
from utils.fairscale_patch import ColumnParallelLinearAG, RowParallelLinearRS
import torch
import torch.nn.functional as F
from torch import nn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from transformers.activations import ACT2FN
from utils.comm import allgather_bsz1


class LlamaMLPTPSP(nn.Module):
    def __init__(self, config, pack_weight: bool = True, sequence_parallel=True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        # TODO(jiarui) use_bias = config.mlp_bias
        use_bias = False
        self.pack_weight = pack_weight
        self.sequence_parallel = sequence_parallel

        if self.pack_weight:
            if sequence_parallel:
                self.w1w3 = ColumnParallelLinearAG(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    bias=use_bias,
                    gather_output=False,
                )
            else:
                self.w1w3 = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    bias=use_bias,
                    gather_output=False,
                )
        else:
            self.w1 = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=use_bias,
                gather_output=False,
            )
            self.w3 = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=use_bias,
                gather_output=False,
            )
        self.w2 = RowParallelLinearRS(
            self.intermediate_size,
            self.hidden_size,
            bias=use_bias,
            input_is_parallel=True,
        )

    def forward(self, x):
        assert (
            self.config.pretraining_tp <= 1
        ), "Pretraining TP is not supported for LlamaMLP"
        if self.pack_weight:
            if not self.sequence_parallel:
                x = allgather_bsz1(x)
            x_packed = self.w1w3(x)
            a1, a3 = x_packed.chunk(2, dim=-1)
            return self.w2(self.act_fn(a1) * a3)
        else:
            x = allgather_bsz1(x)
            return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


def apply_tpsp_attn_patch_llama(model):
    for i in range(model.config.num_hidden_layers):
        new_attn = LlamaFlashAttention2TPSP(
            model.config,
            i,
        ).to(model.dtype)
        new_mlp = LlamaMLPTPSP(model.config).to(model.dtype)
        model.model.layers[i].self_attn = new_attn
        model.model.layers[i].mlp = new_mlp
    print("Applied TP-SP patch for LlamaFlashAttn")
