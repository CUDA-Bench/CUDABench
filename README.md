# CUDABench

**CUDABench** is a comprehensive framework for evaluating Large Language Models (LLMs) on CUDA kernel generation tasks. It provides a complete pipeline from dataset management to LLM code generation and automated performance evaluation on NVIDIA GPUs.

## 📁 Directory Structure

```text
CUDABench/
├── Datasets/           # Contains benchmark tasks (prompts, reference code)
│   └── CUDABench-Set.jsonl
├── Generate/           # LLM inference engine (Multi-API support)
│   ├── main.py
│   ├── config.py
│   └── ...
├── Evaluate/           # Evaluation engine (Compilation & Profiling)
│   ├── manager.py
│   └── ...
└── Results/            # Output directory for generated code
    └── <api_provider>/

```

---

## 🛠 Prerequisites

Before running the benchmark, ensure you have the following installed:

* **Python 3.8+**
* **CUDA Toolkit (`nvcc`)**: Required for compiling generated kernels.
* **NVIDIA Nsight Compute (`ncu`)**: Required for profiling performance metrics.
* **Python Dependencies**:
```bash
pip install openai google-genai anthropic tqdm
```



---

## 🚀 Usage Workflow

### 1. Dataset: CUDABench-Set

The benchmark is driven by `Datasets/CUDABench-Set.jsonl`. Each line in this file is a JSON object representing a unique CUDA task.

**Key Fields per Task:**

* **Metadata**:
* `id`, `task_name`: Unique identifiers for the problem.
* `inputs` / `outputs`: Definitions of tensor shapes and data types (e.g., `float32`, shape `(1048576,)`).


* **Prompts (Difficulty Levels)**:
* `level1_prompt`: **High Detail.** Explicitly describes memory layout, input/output shapes, and algorithmic steps.
* `level2_prompt`: **Standard.** Describes the task conceptually (e.g., "Compute ReLU for array size N").
* `level3_prompt`: **Minimal.** A one-sentence objective (e.g., "Compute ReLU on GPU").


* **Evaluation Artifacts**:
* `bench.cu`: The reference CUDA C++ implementation (Ground Truth).
* `gen.py`: Python script used to generate random binary input data for testing.
* `compare.py`: Python script used to validate the correctness of the generated kernel against the reference.



**Example Entry (Simplified):**

```json
{
  "id": 16,
  "task_name": "ReLU_Activation_Fuction",
  "inputs": [{"name": "relu_input", "dtype": "float32", "shape": "(1048576,)"}],
  "level1_prompt": "...",
  "level2_prompt": "...",
  "level3_prompt": "...",
  "bench.cu": "#include <cuda_runtime.h> ...",
  "gen.py": "import numpy as np ...",
  "compare.py": "def compare_outputs(...) ..."
}
```

### 2. Generate

Use the `Generate` module to query LLMs (DeepSeek, OpenAI, etc.) and generate CUDA kernels.

**Basic Command:**

```bash
python Generate/main.py
```

**With Arguments (Example):**

```bash
python Generate/main.py --api_option deepseek --model_name deepseek-reasoner --samples 3
```

* **Configuration:** You can modify defaults in `Generate/config.py` or pass arguments via CLI.
* **Output:** Generated code is saved to `Results/<api>/<model>_<level>_pass<k>.jsonl`.

### 3. Evaluate

Use the `Evaluate` module to compile, run, and profile the generated code against the reference implementation.

**Command:**

```bash
python Evaluate/manager.py full <path_to_results_file>
```

**Example:**

```bash
python Evaluate/manager.py full Results/deepseek/deepseek-reasoner_level3_pass3.jsonl
```