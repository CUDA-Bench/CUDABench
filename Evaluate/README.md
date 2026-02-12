# Evaluation

This repository provides a **simple, self-contained GPU model evaluation tool** for external users.


## 1. Directory Structure

```
.
├── Evaluate/
│   ├── manager.py
│   ├── eval_from_json.py
│   ├── evaluator_core.py
│   ├── data_process.py
│   └── temp/              # Auto-created, auto-cleaned
├── Datasets/
│   └── CUDABench-Set.jsonl
└── Results/
    └── *.jsonl
```

---

## 2. Requirements

- Python 3.8+
- CUDA Toolkit (nvcc)
- NVIDIA Nsight Compute (ncu)
- Supported NVIDIA GPU

---

## 3. Basic Usage

```bash
python Evaluate/manager.py your_results.jsonl
```

This prints:
- a progress bar
- Pass@1 summary table
- Pass@3 summary table

Nothing else.

---

## 4. Options

### Select GPU

```bash
python Evaluate/manager.py your_results.jsonl --gpu-id 1
```

### Trust mode (skip revalidation)

```bash
python Evaluate/manager.py your_results.jsonl --trust
```

### Explicit dataset path

```bash
python Evaluate/manager.py your_results.jsonl --dataset Datasets/your_dataset.jsonl
```

---

## 5. Output Format

Only **arithmetic mean including zeros** is reported.

```bash
====================================================================================================
PASS@1 SUMMARY | Total Samples: N
====================================================================================================
Pass@ (Correctness):   a/N (xx.xx%)
Pass@ (Functionality): b/N (yy.yy%)
----------------------------------------------------------------------------------------------------
Metric                      | Arith(Inc 0)  
----------------------------------------------------------------------------------------------------
Bandwidth_Utilization       |     xx.xxxx%
Compute_Efficiency          |     xx.xxxx%
Score                       |     xx.xxxx%
====================================================================================================

====================================================================================================
PASS@3 (BEST VERSION) SUMMARY | Total Samples: N
====================================================================================================
Pass@ (Correctness):   a/N (xx.xx%)
Pass@ (Functionality): b/N (yy.yy%)
----------------------------------------------------------------------------------------------------
Metric                      | Arith(Inc 0)  
----------------------------------------------------------------------------------------------------
Bandwidth_Utilization       |     xx.xxxx%
Compute_Efficiency          |     xx.xxxx%
Score                       |     xx.xxxx%
====================================================================================================
```

---

## 6. Notes

- Results JSONL files are automatically searched in `Results/` directory
- Temp files are automatically cleaned before and after each run
- Dataset auto-discovered at `Datasets/CUDABench-Set.jsonl`