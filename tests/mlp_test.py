from utils.globals import _set_global_memory_buffer
import torch
import torch.testing as testing
from decoder.tensor_parallel import LlamaMLPTPSP
from transformers import LlamaConfig
import torch.distributed as dist
import os
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
)


def test_LlamaMLPTPSP():
    dist.init_process_group(backend="nccl")
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    initialize_model_parallel(_world_size)

    dev = torch.device(f"cuda:{_rank}")
    _set_global_memory_buffer(dev)

    intermediate_size = 2048
    hidden_size = 512
    seqlen = 32

    config = LlamaConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    input_tensor = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)
    output_grad = torch.randn(1, seqlen, hidden_size, dtype=torch.bfloat16).to(dev)

    model1 = (
        LlamaMLPTPSP(config, pack_weight=True, sequence_parallel=True)
        .to(torch.bfloat16)
        .to(dev)
    )
    output1 = model1(input_tensor)

    model2 = LlamaMLPTPSP(config, pack_weight=False).to(torch.bfloat16).to(dev)
    model2.w2.weight.data.copy_(model1.w2.weight.data)
    blk_size = intermediate_size // _world_size
    model2.w1.weight.data.copy_(model1.w1w3.weight.data[:blk_size, :])
    model2.w3.weight.data.copy_(model1.w1w3.weight.data[blk_size:, :])
    output2 = model2(input_tensor)

    model3 = (
        LlamaMLPTPSP(config, pack_weight=True, sequence_parallel=True)
        .to(torch.bfloat16)
        .to(dev)
    )
    model3.w2.weight.data.copy_(model1.w2.weight.data)
    model3.w1w3.weight.data.copy_(model1.w1w3.weight.data)
    output3 = model3(input_tensor)

    output1.backward(output_grad)
    output3.backward(output_grad)

    # print(output1 - output2)
    # print(model1.w1w3.weight.grad - model3.w1w3.weight.grad)

    assert torch.allclose(model1.w1w3.weight.grad, model3.w1w3.weight.grad, rtol=1e-3)
    assert torch.allclose(output1, output3, rtol=1e-3)

    assert torch.allclose(output1, output2, rtol=1e-3)


if __name__ == "__main__":
    test_LlamaMLPTPSP()
