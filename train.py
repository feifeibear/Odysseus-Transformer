import argparse
from utils.globals import _set_global_memory_buffer
import torch
import os
from transformers import set_seed
from transformers import AutoModelForCausalLM
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    get_model_parallel_group,
)
import time
import numpy as np

from decoder.odysseus import apply_odysseus_attn_patch_llama
from decoder.ulysses import apply_ulysses_attn_monkey_patch_llama
from decoder.tensor_parallel import apply_tpsp_attn_patch_llama
from decoder.ring import apply_zigzag_ring_attn_patch_llama
from utils.apply_seq_parallel import prepare_attn_inputs


def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    assert args.batch_size == 1, "Only support batch size 1"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    tp_rank = get_model_parallel_rank()
    tp_pg = get_model_parallel_group()

    set_seed(args.seed)
    dev = torch.device(f"cuda:{local_rank}")
    _set_global_memory_buffer(dev)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        rope_theta=args.rope_theta,
        _attn_implementation="flash_attention_2",
        do_sample=True,  # fix warning
        # use_cache=False, # use gradient checkpointing
    )

    if args.parallel_mode == "odysseus":
        apply_odysseus_attn_patch_llama(model)
    elif args.parallel_mode == "ulysses":
        apply_ulysses_attn_monkey_patch_llama()
    elif args.parallel_mode == "tpsp":
        apply_tpsp_attn_patch_llama(model)
    elif args.parallel_mode == "ring":
        apply_zigzag_ring_attn_patch_llama()

    model = model.to(dev)

    if rank == 0:
        print(
            f"{args.parallel_mode} After init model, CUDA memory allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
        )

    assert isinstance(
        model, (transformers.LlamaForCausalLM)
    ), "Only support llama model"

    if world_size > 1:
        if (
            args.parallel_mode == "ulysses"
            or args.parallel_mode == "dp"
            or args.parallel_mode == "ring"
        ):
            model = FSDP(
                model,
                process_group=tp_pg,
                cpu_offload=(
                    CPUOffload(offload_params=True) if args.cpu_offload else None
                ),
            )
        elif args.parallel_mode == "odysseus":
            ignored_modules = []
            layer_num = model.config.num_hidden_layers
            for i in range(layer_num):
                ignored_modules.append(model.model.layers[i].self_attn)
            model = FSDP(
                model,
                ignored_modules=ignored_modules,
                process_group=tp_pg,
                cpu_offload=(
                    CPUOffload(offload_params=True) if args.cpu_offload else None
                ),
            )
        elif args.parallel_mode == "tpsp":
            ignored_modules = []
            layer_num = model.config.num_hidden_layers
            for i in range(layer_num):
                ignored_modules.append(model.model.layers[i].self_attn)
                ignored_modules.append(model.model.layers[i].mlp)
            model = FSDP(
                model,
                ignored_modules=ignored_modules,
                process_group=tp_pg,
                cpu_offload=(
                    CPUOffload(offload_params=True) if args.cpu_offload else None
                ),
            )
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    # if rank == 0:
    #     for name, child in model.named_children():
    #         print(name, child)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, foreach=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True)
    warmup_num_iterations = 2
    elapse = 0.0

    if rank == 0:
        print(
            f"{args.parallel_mode} After init optim, CUDA memory allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
        )
    for step in range(args.max_train_steps):
        if step > warmup_num_iterations:
            start_time = time.time()
        vocab_size = model.config.vocab_size
        batch = torch.randint(vocab_size, size=(1, args.seq_length + 1))

        input_ids = batch[..., :-1]
        target_ids = batch[..., 1:]
        position_ids = (
            torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
        )
        prepared = prepare_attn_inputs(
            args.parallel_mode,
            input_ids,
            position_ids,
            target_ids,
            rank,
            world_size,
            dev,
        )
        local_input_ids = prepared["local_input_ids"]
        local_position_ids = prepared["local_position_ids"]
        local_target_ids = prepared["local_target_ids"]

        logits = model(
            local_input_ids,
            position_ids=local_position_ids,
        ).logits
        loss = loss_func(
            logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
        )
        if rank == 0:
            print(
                f"step {step} CUDA memory allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB, CUDA memory reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
            )
            print(f"loss {loss.item()}")
        loss.backward(loss)

        optim.step()
        optim.zero_grad()

        if step > warmup_num_iterations:
            end_time = time.time()
            elapse += end_time - start_time

        if step >= args.max_train_steps:
            break

    if rank == 0:
        print(f"{args.parallel_mode} Time taken: {elapse:.2f} seconds")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=10)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--rope-theta", type=float, default=100000)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument("--cpu-offload", action="store_true", default=False)
    args.add_argument("--grad-checkpoint", action="store_true", default=False)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["odysseus", "ulysses", "dp", "tpsp", "ring"],
    )
    main(args.parse_args())
