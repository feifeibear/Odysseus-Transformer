import torch


def extract_local_ulysses(value, rank, world_size, device, dim=1):
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]

    return local_value.to(device)


def prepare_ulysses_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):

    local_input_ids = extract_local_ulysses(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local_ulysses(
        position_ids,
        rank,
        world_size,
        device,
    )

    if target_ids is not None:
        local_target_ids = extract_local_ulysses(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }


def prepare_odysseus_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):

    local_input_ids = extract_local_ulysses(
        input_ids,
        rank,
        world_size,
        device,
    )

    if target_ids is not None:
        local_target_ids = extract_local_ulysses(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": position_ids,
        "local_target_ids": local_target_ids,
    }


def prepare_tpsp_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):
    return prepare_odysseus_attn_inputs(
        input_ids, position_ids, target_ids, rank, world_size, device
    )


def prepare_dp_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):
    return {
        "local_input_ids": input_ids.to(device),
        "local_position_ids": position_ids.to(device),
        "local_target_ids": target_ids.to(device),
    }


def extract_local_zigzag_ring(value, rank, world_size, device, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.to(device)


def prepare_zigzag_ring_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):
    local_input_ids = extract_local_zigzag_ring(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local_zigzag_ring(
        position_ids,
        rank,
        world_size,
        device,
    )
    if target_ids is not None:
        local_target_ids = extract_local_zigzag_ring(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids.to(device),
        "local_target_ids": local_target_ids,
    }


def prepare_attn_inputs(parallel_mode, *args, **kwargs):
    if parallel_mode == "ulysses":
        return prepare_ulysses_attn_inputs(*args, **kwargs)
    elif parallel_mode == "odysseus":
        return prepare_odysseus_attn_inputs(*args, **kwargs)
    elif parallel_mode == "tpsp":
        return prepare_tpsp_attn_inputs(*args, **kwargs)
    elif parallel_mode == "dp":
        return prepare_dp_attn_inputs(*args, **kwargs)
    elif parallel_mode == "ring":
        return prepare_zigzag_ring_attn_inputs(*args, **kwargs)
    else:
        raise ValueError(f"Invalid parallel_mode: {parallel_mode}")
