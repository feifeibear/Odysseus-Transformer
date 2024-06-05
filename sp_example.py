import os
import torch
import time

from log_utils import rank_log, get_logger


from llama2_model_sp import SPTransformer, SPModelArgs
from llama2_model_tp import TPTransformer, TPModelArgs
from llama2_model_ulysses import UlyssesSPTransformer, UlyssesSPModelArgs

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    get_model_parallel_group,
)

logger = get_logger()

import argparse

parser = argparse.ArgumentParser(description="PyTorch Distributed Training Example")

# TP-SP: 0.7902135848999023
# SP: 0.38678812980651855
# Ulysses: 0.33405160903930664
parser.add_argument(
    "--parallel_strategy",
    type=str,
    default="Ulysses",
    help="parallel strategy to use (default: SP)",
    choices=["SP", "TP-SP", "Ulysses"],
)
parser.add_argument(
    "--use_profiler",
    action="store_true",
    default=False,
    help="use torch profiler",
)
parser.add_argument(
    "--layer_num",
    type=int,
    default=2,
    help="use torch profiler",
)
parser.add_argument(
    "--seqlen",
    type=int,
    default=128 * 1024,
    help="seqlen",
)

args = parser.parse_args()

parallel_strategy = args.parallel_strategy
layer_num = args.layer_num


def init_prof(use_profiler):
    activities = []
    activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    from contextlib import nullcontext

    ctx = (
        torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/"),
            record_shapes=True,
            with_stack=True,
        )
        if use_profiler
        else nullcontext()
    )
    return ctx


_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])
tp_size = _world_size
assert tp_size == _world_size, f"TP size {tp_size} must match world size {_world_size}"


print(
    f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank} world_size {_world_size}."
)
assert (
    _world_size % tp_size == 0
), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
dp_size = _world_size

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
initialize_model_parallel(_world_size)
tp_rank = get_model_parallel_rank()
tp_pg = get_model_parallel_group()

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
if parallel_strategy == "SP":
    model_args = SPModelArgs(dim=256, n_layers=layer_num, n_heads=16, vocab_size=32000)
    model = SPTransformer(model_args).to(torch.bfloat16).to(f"cuda:{tp_rank}")
elif parallel_strategy == "TP-SP":
    model_args = TPModelArgs(dim=256, n_layers=layer_num, n_heads=16, vocab_size=32000)
    model = TPTransformer(model_args).to(torch.bfloat16).to(f"cuda:{tp_rank}")
elif parallel_strategy == "Ulysses":
    model_args = UlyssesSPModelArgs(
        dim=256, n_layers=layer_num, n_heads=16, vocab_size=32000
    )
    model = UlyssesSPTransformer(model_args).to(torch.bfloat16).to(f"cuda:{tp_rank}")
else:
    raise TypeError(f"Invalid parallel strategy {parallel_strategy}")

# init model weights
model.init_weights()

if parallel_strategy == "SP":
    for i in range(layer_num):
        model.layers[i].feed_forward = FSDP(
            model.layers[i].feed_forward, process_group=tp_pg
        )
elif parallel_strategy == "Ulysses":
    model = FSDP(model, process_group=tp_pg)

rank_log(_rank, logger, f"Model after parallelization {model=}\n")

# Create an optimizer for the parallelized and sharded model.
lr = 3e-3
rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

# Training loop:
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
rank_log(_rank, logger, f"\nStarting {parallel_strategy} training...")
num_iterations = 20
seqlen = args.seqlen
bs = 1

warmup_num_iterations = 2
elapse = 0

assert bs == 1, f"Batch size {bs} must be 1 for this test"
ctx = init_prof(args.use_profiler)

with ctx as prof:
    for i in range(num_iterations):
        # seeding with tp_rank to ensure identical inputs for TP groups
        if i > warmup_num_iterations:
            start_time = time.time()
        torch.manual_seed(i)
        inp = torch.randint(32000, (bs, seqlen), device=f"cuda:{tp_rank}")
        if parallel_strategy == "Ulysses":
            inp = inp.chunk(tp_size, dim=1)[tp_rank]

        output = model(inp)

        output.sum().backward()

        optimizer.step()
        optimizer.zero_grad()
        # rank_log(_rank, logger, f"2D iter {i} complete")
        if i > warmup_num_iterations:
            end_time = time.time()
            elapse += end_time - start_time

        if args.use_profiler:
            prof.step()

rank_log(
    _rank,
    logger,
    f"{parallel_strategy}: elapse {elapse:.2f} sec successfully completed!",
)
