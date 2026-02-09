# Evaluation

This repository provides a **simple, self-contained GPU model evaluation tool** for external users.


## 1. Directory Structure

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

## 2. Requirements

- Python 3.8+
- CUDA Toolkit (nvcc)
- NVIDIA Nsight Compute (ncu)
- Supported NVIDIA GPU

---

## 3. Basic Usage

```bash
python scripts/manager.py full your_results.jsonl
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
python scripts/manager.py full your_results.jsonl --gpu-id 1
```


### Explicit dataset path

```bash
python scripts/manager.py full your_results.jsonl --dataset datasets/your_dataset.jsonl
```

---

## 5. Output Format

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