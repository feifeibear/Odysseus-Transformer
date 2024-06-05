## Odysseus: Upgrade DeepSpeed-Ulysses by Decoupling the Parallel Strategies of Attention and FFN

This repository delves into the optimal parallelization strategies for long-sequence LLMs, implementing four methods: 
1. [Tensor Parallelism (TP)](https://arxiv.org/abs/1909.08053), the classical implementation proposed by Megatron-LM, using AllReduce for activation communications.
2. [Tensor Parallelism with Sequence Parallelism (TP-SP)](https://arxiv.org/abs/2205.05198), refer to MLSys 23' paper: Reducing Activation Recomputation in Large Transformer Models.
3. [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509), refer to paper: DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models.
4. Odysseus. A novel method proposed in this repo.

As you can see, we involve both tensor parallelism and sequence parallelism here.
The commonality among these methods is that they all require partitioning along the head number dimension, thus the degree of parallelism is constrained by the head number. We have excluded Ring-Attention as it can be used orthogonally with these four methods.

Odysseus, our novel sequence parallel approach, decouples the parallelization of Attention and FFN components within Transformers. Attention leverages TP-SP, partitioning QKVO weights and employing Allgather for hidden state inputs and output tensors, with ReduceScatter communication, segmenting Activation along the sequence dimension. FFN adopts sequence parallelism, dividing input by sequence dimension without Activation communication, focusing on gradient communication for parameters during backpropagation.

The communication and memory costs of these four methods are summarized in the table below. Among them, RS stands for ReduceScatter, and AG stands for AllGather. L represents the sequence length, d is the hidden dimension, i is the intermediate hidden size, with GPT-4 having i = 4d, and N denotes the number of GPUs.

When the sequence length $L$ exceeds the intermediate hidden size $i$ ($L$ > i), Odysseus+ZeRO3 demonstrates lower communication cost compared to TP-SP and Ulysses+ZeRO3. Notably, all three methods maintain similar memory consumption.

| Method          | Comm Activation | Comm Volume       | Comm Gradient | Comm Volume                   | Mem Activation | Mem Param/Grad |
|-----------------|------------|--------------|----------|--------------------------|------------|------------|
| TP              | 2AllReduce | 4O(Ld)       | 0        | 0                        | full       | 1/N        |
| TP-SP           | 4RS+4AG    | 4O(Ld)       | 0        | 0                        | 1/N        | 1/N        |
| Ulysses+ZeRO3   | 4All2All   | 4O(Ld)       | AllReduce| 4O($d^2$)+2O(di)           | 1/N        | 1/N      |
| Odysseus+ZeRO3  | 2RS+2AG    | 2O(Ld)       | AllReduce (FFN) | 2O(di) | 1/N        | 1/N        |

### Usage
1. Install requirements.txt
2. Install [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention)
3. bash run_example.sh

### TODO:
The repo is work in progress.

1. Our TP-SP implementation stores the full shape tensor after allgather in GPU memory before the backward pass, resulting in an Activation memory cost greater than 1/N. Our implementation does not strictly adhere to the paper.

2. Integrate Odysseus with Ring for hybrid parallelism.


### Acknowledgements
The repo is built on [pytorch/example](https://github.com/pytorch/examples).
