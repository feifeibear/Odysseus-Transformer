
# To run samples:
# bash run_example.sh {file_to_run.py} {num_gpus}
# where file_to_run = example to launch.  Default = 'fsdp_tp_example.py'
# num_gpus = num local gpus to use (must be at least 2). Default = 4

# samples to run include:
# sequence_parallel_example.py
# tensor_parallel_example.py
# fsdp_tp_example.py

SCRIPT="sp_example.py"

# docker exec -it 888c58e74578 bash
GPU_NUM=8
echo "Launching with ${GPU_NUM} gpus"
torchrun --nnodes=1 --nproc_per_node=${GPU_NUM} --rdzv_id=101 --rdzv_endpoint="localhost:5972" $SCRIPT --parallel_strategy "Ulysses"
# torchrun --nnodes=1 --nproc_per_node=${GPU_NUM} --rdzv_id=101 --rdzv_endpoint="localhost:5972" $SCRIPT --parallel_strategy "SP"
# torchrun --nnodes=1 --nproc_per_node=${GPU_NUM} --rdzv_id=101 --rdzv_endpoint="localhost:5972" $SCRIPT --parallel_strategy "TP-SP"
# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="localhost:5972" test_ring.py