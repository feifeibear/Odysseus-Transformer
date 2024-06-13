from decoder.odysseus import LlamaFlashAttention2TPSP
from utils.fairscale_patch import RowParallelLinearRS
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        # TODO(jiarui) use_bias = config.mlp_bias
        use_bias = False

        self.w1 = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size, bias=use_bias, gather_output=False
        )
        self.w2 = RowParallelLinearRS(
            self.intermediate_size,
            self.hidden_size,
            bias=use_bias,
            input_is_parallel=True,
        )
        self.w3 = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size, bias=use_bias, gather_output=False
        )

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining TP is not supported for LlamaMLP")
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
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
