# SEQLEN=16384
SEQLEN=32768
# SEQLEN=49152
# SEQLEN=65536
export CUDA_VISIBLE_DEVICES=4,5,6,7
GPU_NUM=4

PARALLEL="odysseus"
CPUOFFLOAD="--cpu-offload"
GRADCKP="--grad-checkpoint"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP

PARALLEL="ulysses"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP

PARALLEL="tpsp"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/fjr/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL $CPUOFFLOAD $GRADCKP
