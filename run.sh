# SEQLEN=16384
# SEQLEN=32768
SEQLEN=49152
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPU_NUM=8

PARALLEL="odysseus"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/shangchun/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL

PARALLEL="ulysses"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/shangchun/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL

PARALLEL="tpsp"
torchrun --nproc_per_node=$GPU_NUM train.py --model /mnt/shangchun/Llama-2-7b-hf --seq-length $SEQLEN --parallel_mode $PARALLEL
