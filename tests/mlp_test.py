from utils.globals import _set_global_memory_buffer
import torch
from decoder.tensor_parallel import LlamaMLPTPSP
from decoder.odysseus import LlamaMLPZeRO
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers import LlamaConfig
import torch.distributed as dist
import os
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    get_model_parallel_group,
)
import time


def test_LlamaMLPTPSP(intermediate_size=2048, hidden_size=512, seqlen=32):
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    dev = torch.device(f"cuda:{_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        initialize_model_parallel(_world_size)
        _set_global_memory_buffer(dev)

    config = LlamaConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    input_tensor = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)
    output_grad = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)

    # packed TPSP baseline
    model1 = (
        LlamaMLPTPSP(
            config,
            pack_weight=True,
            sequence_parallel=False,
            keep_master_weight_for_test=True,
        )
        .to(torch.bfloat16)
        .to(dev)
    )
    output1 = model1(input_tensor)
    output1.backward(output_grad)

    # unpacked TPSP
    model2 = LlamaMLPTPSP(config, pack_weight=False).to(torch.bfloat16).to(dev)
    model2.w2.weight.data.copy_(model1.w2.weight.data)
    blk_size = intermediate_size // _world_size
    model2.w1.weight.data.copy_(model1.w1w3.weight.data[:blk_size, :])
    model2.w3.weight.data.copy_(model1.w1w3.weight.data[blk_size:, :])
    output2 = model2(input_tensor)
    output2.backward(output_grad)

    # packed + megatron-LM Function TPSP
    model3 = (
        LlamaMLPTPSP(config, pack_weight=True, sequence_parallel=True)
        .to(torch.bfloat16)
        .to(dev)
    )
    model3.w2.weight.data.copy_(model1.w2.weight.data)
    model3.w1w3.weight.data.copy_(model1.w1w3.weight.data)
    output3 = model3(input_tensor)
    output3.backward(output_grad)

    # zero3
    model4 = LlamaMLPZeRO(config).to(torch.bfloat16).to(dev)
    w1w3 = torch.cat(
        [model2.w1.get_master_weight().data, model2.w3.get_master_weight().data], dim=0
    )
    sharded_w1w3 = w1w3.chunk(_world_size, dim=0)[_rank]
    model4.gate_up_proj.weight.data.copy_(sharded_w1w3.data)

    shard_w2 = model2.w2.get_master_weight().data.chunk(_world_size, dim=0)[_rank]
    model4.down_proj.weight.data.copy_(shard_w2)

    output4 = model4(input_tensor)
    output4.backward(output_grad)

    # print(torch.max(output1 - output4))
    # TODO(jiarui) error is large
    assert torch.allclose(output1, output4, atol=1e-1)

    # print(model1.w1w3.weight.grad - model3.w1w3.weight.grad)

    assert torch.allclose(model1.w1w3.weight.grad, model3.w1w3.weight.grad, rtol=1e-3)
    assert torch.allclose(output1, output3, rtol=1e-3)

    assert torch.allclose(output1, output2, rtol=1e-3)


def benchmark_LlamaMLPTPSP(
    intermediate_size, hidden_size, seqlen, pack_weight, sequence_parallel
):
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    dev = torch.device(f"cuda:{_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        initialize_model_parallel(_world_size)
        _set_global_memory_buffer(dev)

    config = LlamaConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    input_tensor = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)
    output_grad = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)

    model1 = (
        LlamaMLPTPSP(
            config, pack_weight=pack_weight, sequence_parallel=sequence_parallel
        )
        .to(torch.bfloat16)
        .to(dev)
    )

    output = model1(input_tensor)
    output.backward(output_grad)

    iter_num = 10
    start1 = time.time()

    for i in range(iter_num):
        input_tensor = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)
        output1 = model1(input_tensor)
        output1.backward(output_grad)
    end1 = time.time()
    elapsed1 = end1 - start1

    if _rank == 0:
        print(
            f"LlamaMLPTPSP seqlen {seqlen/1024}K, pack weight {pack_weight} sequence parallel {sequence_parallel} elapsed:: {elapsed1:.2f} sec"
        )
        print(
            f"CUDA memory allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
        )


def benchmark_LlamaMLPFSDP(intermediate_size, hidden_size, seqlen):
    import os

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    dev = torch.device(f"cuda:{_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        initialize_model_parallel(_world_size)
        _set_global_memory_buffer(dev)

    pg = get_model_parallel_group()

    config = LlamaConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    input_tensor = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)
    output_grad = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)

    model1 = LlamaMLP(config).to(torch.bfloat16).to(dev)

    model1 = FSDP(model1, process_group=pg)
    output = model1(input_tensor)
    output.backward(output_grad)

    iter_num = 10
    start1 = time.time()
    for i in range(iter_num):
        input_tensor = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)
        output1 = model1(input_tensor)
        output1.backward(output_grad)
    torch.cuda.synchronize()
    end1 = time.time()
    elapsed1 = end1 - start1

    if _rank == 0:
        print(f"LlamaMLPFSDP seqlen {seqlen/1024}K elapsed: {elapsed1:.2f} sec")
        print(
            f"CUDA memory allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
        )


# torchrun --nproc_per_node=8 ./tests/mlp_test.py
if __name__ == "__main__":

    hidden_size = 4096
    intermediate_size = 11008
    seqlen = 4 * 1024

    for seqlen in [8192]:
        test_LlamaMLPTPSP(intermediate_size, hidden_size, seqlen)
        # benchmark_LlamaMLPFSDP(intermediate_size, hidden_size, seqlen)
        # benchmark_LlamaMLPTPSP(
        #     intermediate_size,
        #     hidden_size,
        #     seqlen,
        #     pack_weight=True,
        #     sequence_parallel=True,
        # )
        # benchmark_LlamaMLPTPSP(
        #     intermediate_size,
        #     hidden_size,
        #     seqlen,
        #     pack_weight=False,
        #     sequence_parallel=False,
        # )
        # benchmark_LlamaMLPTPSP(
        #     intermediate_size,
        #     hidden_size,
        #     seqlen,
        #     pack_weight=True,
        #     sequence_parallel=False,
        # )
        print("\n")
