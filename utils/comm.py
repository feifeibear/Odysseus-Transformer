from .fairscale_patch import allgather_along_first_dim_from_model_parallel_region


def allgather_bsz1(x):
    bsz, seqlen, h = x.shape
    assert bsz == 1, f"Batch size {bsz} must be 1 for allgather_bsz1"
    # NOTE gather along the last dim
    x = allgather_along_first_dim_from_model_parallel_region(x)
    x = x.view(bsz, -1, h)
    return x
