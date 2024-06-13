import torch
from functools import reduce
import operator

_GLOBAL_MEMORY_BUFFER = None


# copy from megatron-lm
class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self, dev):
        self.buffer = {}
        self.dev = dev

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=self.dev, requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _set_global_memory_buffer(dev):
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer(dev)


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, "global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER
