# LOCAL_SEQLEN=1024
# LOCAL_SEQLEN=6144
LOCAL_SEQLEN=8192
# LOCAL_SEQLEN=16384
# LOCAL_SEQLEN=32768
# LOCAL_SEQLEN=49152
# LOCAL_SEQLEN=65536
# LOCAL_SEQLEN=98304
# export CUDA_VISIBLE_DEVICES=4,5,6,7

GPU_NUM=8
SEQLEN=$((GPU_NUM*LOCAL_SEQLEN))

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TF_CPP_MIN_LOG_LEVEL=2
# FSDP https://github.com/pytorch/pytorch/issues/124260
export TORCH_NCCL_AVOID_RERORD_STREAMS=1
# CPUOFFLOAD="--cpu-offload"

# not ready
# GRADCKP="--grad-checkpoint"

PARALLEL="ulysses"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP

PARALLEL="ring"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP


PARALLEL="odysseus"
env PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' \
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf \
--seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP --use_ody_zero3


export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
PARALLEL="tpsp"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP
 