# GPU Model Evaluation Tool 

This repository provides a **simple, self-contained GPU model evaluation tool** for external users.

The goal is straightforward:

> Run one command, get Pass@1 / Pass@3 accuracy and efficiency statistics, printed directly in the terminal, with zero side effects.

No intermediate files are kept.
No manual directory management is required.

---

## Directory Structure

```
.
├── scripts/
│   ├── manager.py
│   ├── eval_from_json.py
│   ├── evaluator_core.py
│   └── data_process.py
├── datasets/
│   └── *.jsonl
└── *.jsonl
```

---

## Requirements

- Python 3.8+
- CUDA Toolkit (nvcc)
- NVIDIA Nsight Compute (ncu)
- Supported NVIDIA GPU

---

## Basic Usage

```bash
python scripts/manager.py full your_results.jsonl
```

This prints:
- a progress bar
- Pass@1 summary table
- Pass@3 summary table

Nothing else.

---

## Options

### Select GPU

```bash
python scripts/manager.py full your_results.jsonl --gpu-id 1
```


### Explicit dataset path

```bash
python scripts/manager.py full your_results.jsonl --dataset datasets/your_dataset.jsonl
```

---

## Output Format

Only **arithmetic mean including zeros** is reported.

```bash
====================================================================================================
BENCHMARK REPORT (PASS@1) | Total Samples: N
====================================================================================================
Pass@ (Correctness):   a/N (xx.xx%)
Pass@ (Functionality): b/N (yy.yy%)
----------------------------------------------------------------------------------------------------
Metric                     | Arith(Inc 0)
----------------------------------------------------------------------------------------------------
Bandwidth_Utilization      |   xx.xxxx%
Compute_Efficiency         |    x.xxxx%
Score                      |   xx.xxxx%
====================================================================================================
```

---

## Zero Side-Effect Guarantee

All intermediate files are created in a system temporary directory and automatically deleted after execution.

Your project directory is never modified.

---

## Summary

- One command
- One input file
- Two final tables
- No leftover files

Designed for clean external use.