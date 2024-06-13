# SEQLEN=1024
SEQLEN=32768
# SEQLEN=49152
# SEQLEN=65536
# SEQLEN=98304
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPU_NUM=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

# CPUOFFLOAD="--cpu-offload"

# not compatible with TP-SP and Odysses
# GRADCKP="--grad-checkpoint"


PARALLEL="ulysses"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP

PARALLEL="ring"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP


PARALLEL="odysseus"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP

PARALLEL="tpsp"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP
