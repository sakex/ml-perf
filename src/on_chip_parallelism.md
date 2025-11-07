# On-Chip Parallelism

Machine Learning workloads require more and more computational power as we scale the number of parameters, the context lengths, and the amount of data we ingest. At the same time, chip design has hit a plateau; it is getting prohibitively expensive to increase the number of operations a chip can do per second. Furthermore, memory latency and bandwidth have not been keeping up with the increases in compute speed, implying that computational power cannot be fully leveraged because the data cannot be moved as fast as it is being processed. We cannot rely on faster chips, so we instead rely on **the chips doing more at the same time** either by doing multiple operations at once or having multiple cores working together in parallel.

## Sequential execution model

[Traditional chips were thought of as having two main blocks](https://en.wikipedia.org/wiki/Von_Neumann_architecture); the memory (RAM) and the Central Processing Unit (CPU.)

Traditional software is usually written with this implied model:

1. Load some scalars from RAM to the CPU
2. Do some operations on the CPU
3. Write back the output of those operations to RAM
4. Repeat for the next instruction

While this model is great and allowed us to write most of the software running the world today; it has long become incoherent with the way chips actually process data. We let the compilers and the chips themselves rewrite our code to make better use of the actual capabilities of the hardware; mostly through different levels of parallelism and better memory access patterns.

## The different levels of on-chip parallelism

Modern chips are all inherently parallel. Whether they are GPUs, TPUs, or modern CPUs. They also all feature different types of parallelism that need to be exploited to maximize the chip's utilisation. Exploiting these mechanisms is not always explicit because compilers are reasonably good at leveraging target architectures's features. Some chips are also capable of rewriting the machine code they receive before executing it.

### IO parallelism

The processing unit and the memory are two independent units. Therefore, the processor is able to perform computations independently of the memory reads and writes. For instance, it can request some data from RAM as well as perform an addition between two numbers it has already loaded while waiting for the data to be received. This means we can potentially completely overlap computation times with memory movements. In our execution model, steps 1, 2, and 3 can all be executed in parallel.

### Single Instruction Multiple Data ([SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data))

Most modern chips are capable of executing a single instruction on multiple elements at once. This can mean adding two vectors with one another in one cycle, reducing (ie. summing) a vector into a scalar, or even running a matrix dot product within specialized Arithmetic Logic Units (ALUs) in TPUs' and GPUs' tensor cores.

- Modern x86 chips feature AVX registers
- TPUs have MXUs for matrix dot products, VPUs for elementwise vectorized operations, and XLUs for reductions
- GPUs have tensor cores for matrix dot products

Coming back to our execution model. Instead of executing one operation at a time on scalars, we instead perform as many operations as we can in parallel within a SIMD unit and also load more data at once since our registers are larger.

### Instruction Level Parallelism

As we have mentioned, modern chips possess multiple circuits that specialize in the handling of different data types and operations. For instance, TPUs have MXUs and VPUs. Some of these circuits can also be used independently. For instance, we could compute a dot product on the MXU, and apply a ReLU activation at the same time on the VPU (more specifically, perform a dot product, write the output to the VPU, do the next dot product at the same time as we apply ReLU on the VPU.)

### Multiple threads of execution

Finally, modern architectures usually feature multiple processing units that can execute operations independently of one another. This is the main differentiator of GPUs which possess thousands of cores that can all execute operations in parallel on different data addresses. This model comes with additional complexities such as the need to synchronize data across cores safely and efficiently.

Coming back to the original model, we now execute the model multiple times in parallel.

## Comparison of On-Chip Parallelism

| Parallelism Type | Core Concept | ⚙️ Hardware Example | 💻 Software Abstraction | 👤 Who Implements This? |
| :--- | :--- | :--- | :--- | :--- |
| **IO parallelism** | Hiding memory latency by performing computation while waiting for data to be fetched. | GPU warp schedulers swapping threads stalled on memory reads; hardware prefetchers. | Optimized kernels (e.g., in cuDNN, XLA). | **Chip Hardware** (schedulers) & **Compiler** (instruction scheduling). |
| **SIMD** <br/> (Single Instruction, Multiple Data) | One instruction operating on many data elements (a vector) at once. | GPU Tensor Cores (for matrices), TPU MXUs, CPU AVX registers. | Vectorized code (e.g., `a + b` on tensors), `torch.matmul`. | **Compiler / Library** (e.g., cuDNN, XLA). The programmer *enables* this by using high-level vector/matrix ops. |
| **Instruction-Level Parallelism** <br/> (Using Multiple ALUs) | Using different, specialized execution units (ALUs) within a core at the same time. | A TPU pipelining work from its MXU (matrix) to its VPU (vector). | **Kernel Fusion** (e.g., `matmul + relu` in one operation). | **Compiler** (e.g., `jax.jit`, XLA). The chip hardware makes it possible. |
| **Multithreading / Multicore** <br/> (MIMD / SIMT) | Multiple processing units (cores) executing instructions independently. | Multi-core CPU (MIMD), thousands of CUDA Cores on a GPU (SIMT). | **Data Parallelism** (splitting a batch over cores) or Model Parallelism. | **Programmer & Library** (e.g., CUDA, which manages threads for kernels). |