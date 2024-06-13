# test.py
import torch
import torch.distributed as dist
from decoder.odysseus import LlamaFlashAttention2TPSP, LlamaFlashAttention2
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
)
from fairscale.nn.model_parallel.mappings import (
    gather_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
import os
from transformers import LlamaConfig
from utils.comm import allgather_bsz1


def compare_forward_results():
    dist.init_process_group(backend="nccl")
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    hidden_size = 4096
    seqlen = 10
    bsz = 1
    dtype = torch.bfloat16

    initialize_model_parallel(_world_size)

    torch.manual_seed(_rank)
    input = torch.randn(bsz, seqlen, hidden_size, dtype=dtype).to(dist.get_rank())

    # (bsz, seqlen, hidden_size)
    position_ids_1 = (
        torch.arange(seqlen * _world_size)
        .unsqueeze(0)
        .repeat(bsz, 1)
        .to(dist.get_rank())
    )

    # position_ids_2 = position_ids_1.chunk(_world_size, dim = 1)[get_model_parallel_rank()]

    config = LlamaConfig(
        hidden_size=hidden_size, num_attention_heads=32, num_key_value_heads=8
    )
    model1 = LlamaFlashAttention2TPSP(config).to(dtype).to(dist.get_rank())
    model2 = LlamaFlashAttention2(config).to(dtype).to(dist.get_rank())

    def copy_weight(t1, t2):
        t1.data.copy_(t2.data)

    copy_weight(model2.q_proj.weight, model1.q_proj.get_master_weight())
    copy_weight(model2.k_proj.weight, model1.k_proj.get_master_weight())
    copy_weight(model2.v_proj.weight, model1.v_proj.get_master_weight())
    copy_weight(model2.o_proj.weight, model1.o_proj.get_master_weight())

    output1, _, _ = model1(input, position_ids=position_ids_1)
    output1_global = allgather_bsz1(output1)
    # output1_global = output1

    print(f"input {_rank}: {input}")
    input_global = allgather_bsz1(input)
    print(f"input_global {_rank}: {input_global}")
    output2_global, _, _ = model2(input_global, position_ids=position_ids_1)

    # print(f"output1_global {_rank}: {output1_global}")
    # print(f"output2_global {_rank}: {output2_global}")
    print(f"{torch.max(output2_global - output1_global)}")
    if torch.allclose(output1_global, output2_global, atol=1e-1, rtol=1e-1):
        print(f"rank {_rank}: The forward results are the same.")
    else:
        print(f"rank {_rank}: The forward results are different.")


if __name__ == "__main__":
    compare_forward_results()
