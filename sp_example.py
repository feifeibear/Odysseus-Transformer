import os
import torch
import time

from log_utils import rank_log, get_logger


from llama2_model_sp import SPTransformer, SPModelArgs
from llama2_model_tp import TPTransformer, TPModelArgs 

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    get_model_parallel_group
)

logger = get_logger()

import argparse

parser = argparse.ArgumentParser(description='PyTorch Distributed Training Example')

parser.add_argument('--parallel_strategy', type=str, default='SP',
                    help='parallel strategy to use (default: SP)',
                    choices=['SP', 'TP-SP'])

args = parser.parse_args()

parallel_strategy = args.parallel_strategy

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])
tp_size = _world_size
assert tp_size == _world_size, f"TP size {tp_size} must match world size {_world_size}"


print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank} world_size {_world_size}.")
assert (
    _world_size % tp_size == 0
), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
dp_size = _world_size // tp_size

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
initialize_model_parallel(_world_size)
tp_rank = get_model_parallel_rank()
tp_pg = get_model_parallel_group()

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
if parallel_strategy == "SP":
    model_args = SPModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)
    model = SPTransformer(model_args).to(f"cuda:{tp_rank}")
    # def custom_auto_wrap_policy(
    #     module: nn.Module,
    #     recurse: bool,
    #     nonwrapped_numel: int,
    #     ) -> bool:
    #         if "feed_forward" in module.__class__.__name__:
    #             # print(f"fsdp: module {module.__class__.__name__}")
    #             return True
    #         else:
    #             print(f"fsdp: module {module.__class__.__name__}")
    #             return False
    # my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, recurse = True)
elif parallel_strategy == "TP-SP":
    model_args = TPModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)
    model = TPTransformer(model_args).to(f"cuda:{tp_rank}")
                                         

# init model weights
model.init_weights()

if parallel_strategy == "SP":
    model.layers[0].feed_forward = FSDP(model.layers[0].feed_forward)
    model.layers[1].feed_forward = FSDP(model.layers[1].feed_forward)


    
rank_log(_rank, logger, f"Model after parallelization {model=}\n")

# Create an optimizer for the parallelized and sharded model.
lr = 3e-3
rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

# Training loop:
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
rank_log(_rank, logger, "\nStarting 2D training...")
num_iterations = 20    
seqlen = 32*1024
bs = 1

warmup_num_iterations = 2
elapse = 0

assert bs == 1, f"Batch size {bs} must be 1 for this test"

for i in range(num_iterations):
    # seeding with tp_rank to ensure identical inputs for TP groups
    if i > warmup_num_iterations:
        start_time = time.time()
    torch.manual_seed(i)
    inp = torch.randint(32000, (bs, seqlen), device=f"cuda:{tp_rank}")

    output = model(inp)
    output.sum().backward()

    optimizer.step()
    optimizer.zero_grad()
    rank_log(_rank, logger, f"2D iter {i} complete")
    if i > warmup_num_iterations:
        end_time = time.time() 
        elapse += end_time - start_time

rank_log(_rank, logger, f"{parallel_strategy}: elapse {elapse} successfully completed!")
