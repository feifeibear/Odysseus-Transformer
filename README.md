## Odysseus: Sequence Parallelism by Decoupling Attention and FFN for Enhanced LLM Performance

This repository delves into the optimal parallelization strategies for long-sequence LLMs, implementing four methods: Tensor Parallelism (TP), Tensor Parallelism with Sequence Parallelism (TP-SP), DeepSpeed-Ulysses, and Odysseus. The commonality among these methods is that they all require partitioning along the head number dimension, thus the degree of parallelism is constrained by the head number. We have excluded Ring-Attention as it can be used orthogonally with these four methods.

Odysseus, our novel sequence parallel approach, decouples the parallelization of Attention and FFN components within Transformers. Attention leverages TP-SP, partitioning QKVO weights and employing Allgather for hidden state inputs and output tensors, with ReduceScatter communication, segmenting Activation along the sequence dimension. FFN adopts sequence parallelism, dividing input by sequence dimension without Activation communication, focusing on gradient communication for parameters during backpropagation.

The communication and memory overheads of these four methods are summarized in the table below. Among them, RS stands for ReduceScatter, and AG stands for AllGather. L represents the sequence length, d is the hidden dimension, i is the intermediate hidden size, with GPT-4 having i = 4d, and N denotes the number of GPUs.



| Method          | Comm Activation | Comm Volume       | Comm Gradient | Comm Volume                   | Mem Activation | Mem Param/Grad |
|-----------------|------------|--------------|----------|--------------------------|------------|------------|
| TP              | 2AllReduce | 4O(Ld)       | 0        | 0                        | full       | 1/N        |
| TP-SP           | 4RS+4AG    | 4O(Ld)       | 0        | 0                        | 1/N        | 1/N        |
| Ulysses+ZeRO3   | 4All2All   | 4O(Ld)       | AllReduce| 4O($d^2$)+2O(di)           | 1/N        | 1/N      |
| Odysseus+ZeRO3  | 2RS+2AG    | 2O(Ld)       | AllReduce (FFN) | 2O(di) | 1/N        | 1/N        |

