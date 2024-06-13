from utils.comm import allgather_bsz1
import transformers
from typing import List, Optional, Tuple, Union
import warnings
import torch
import torch.utils.checkpoint

try:
    from yunchang.ulysses import UlyssesAttention

    ulysses_attn = UlyssesAttention()
except:
    print(
        "If you want to use the UlyssesAttention class, please install the yunchang package."
    )
    ulysses_attn = None

import fairscale.nn.model_parallel.initialize as fs_init
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    apply_rotary_pos_emb,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from utils.fairscale_patch import RowParallelLinearRS

logger = logging.get_logger(__name__)


class NewLlamaFlashAttention2(LlamaFlashAttention2):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, keep_master_weight_for_test=False, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        self.model_parallel_size = fs_init.get_model_parallel_world_size()
        self.local_rank = fs_init.get_model_parallel_rank()

        self.n_local_heads = self.num_heads // self.model_parallel_size
        self.n_local_kv_heads = self.num_key_value_heads // self.model_parallel_size

        # clear allocated memory
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            keep_master_weight_for_test=keep_master_weight_for_test,
        )

        # self.wq.weight.copy(self.q_proj.weight.chunk(self.model_parallel_size, dim=1)[self.local_rank])
        # if self.wq.bias:
        #     self.wq.bias.copy(self.q_proj.bias.chunk(self.model_parallel_size, dim=1)[self.local_rank])
        # del self.q_proj

        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            keep_master_weight_for_test=keep_master_weight_for_test,
        )

        # self.wk.weight.copy(self.k_proj.weight.chunk(self.model_parallel_size, dim=1)[self.local_rank])
        # if self.wk.bias:
        #     self.wk.bias.copy(self.k_proj.bias.chunk(self.model_parallel_size, dim=1)[self.local_rank])
        # del self.k_proj

        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            keep_master_weight_for_test=keep_master_weight_for_test,
        )

        # self.wv.weight.copy(self.v_proj.weight.chunk(self.model_parallel_size, dim=1)[self.local_rank])
        # if self.wv.bias:
        #     self.wv.bias.copy(self.v_proj.bias.chunk(self.model_parallel_size, dim=1)[self.local_rank])
        # del self.v_proj

        self.o_proj = RowParallelLinearRS(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            keep_master_weight_for_test=keep_master_weight_for_test,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False
        # local
        bsz, local_q_len, _ = hidden_states.size()
        assert bsz == 1, f"bsz: {bsz}"
        hidden_states = allgather_bsz1(hidden_states)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.n_local_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_local_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_local_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def apply_odysseus_attn_patch_llama(model):
    for i in range(model.config.num_hidden_layers):
        new_module = NewLlamaFlashAttention2(
            model.config,
            i,
        ).to(model.dtype)
        model.model.layers[i].self_attn = new_module
    print("Applied Odysseus patch for LlamaFlashAttn")
