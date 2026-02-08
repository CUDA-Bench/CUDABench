#!/usr/bin/env python3
"""
evaluator_core.py - Evaluation core module (reused by model eval / bench eval).

Notes:
- Extracted common evaluation utilities from eval_from_json.py to reduce duplication.
- Not tied to a specific results input format; upper-layer scripts decide where code comes from.
"""

from __future__ import annotations

import os
import io
import contextlib
import subprocess
import shutil
from pathlib import Path
import pandas as pd


# ============================================================
# Working Directory Context & Run Script
# ============================================================
@contextlib.contextmanager
def working_directory(path: str):
    """               """
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def run_script_as_function(
    code_str: str,
    work_dir: str,
    extra_globals: dict | None = None,
    capture_stdout: bool = True,
):
    """
            Python       exec

    Returns:
        (success: bool, output: str)
    """
    globals_dict = {"__name__": "__main__"}
    if extra_globals:
        globals_dict.update(extra_globals)

    buffer = io.StringIO()
    try:
        with working_directory(work_dir):
            if capture_stdout:
                with contextlib.redirect_stdout(buffer):
                    exec(code_str, globals_dict)
            else:
                exec(code_str, globals_dict)
        return True, buffer.getvalue()
    except Exception as e:
        return False, str(e)


# ============================================================
#    CUDA
# ============================================================
def compile_code(code: str, work_dir: str, code_filename: str = "kernel.cu"):
    """   CUDA                     None"""
    cu_file_path = os.path.join(work_dir, code_filename)
    executable_path = os.path.join(work_dir, "kernel")

    with open(cu_file_path, "w") as f:
        f.write(code)

    compile_command = ["nvcc", cu_file_path, "-o", executable_path]
    try:
        subprocess.run(
            compile_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            text=True,
        )
        return executable_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def run_exe(exe_path: str, work_dir: str, timeout: int = 60):
    """

    Returns:
        bool: returncode==0
    """
    try:
        exe_file = "./" + os.path.basename(exe_path)
        result = subprocess.run(
            [exe_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            cwd=work_dir,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


# ============================================================
# Correctness / Functionality    revalidate
# ============================================================
def evaluate_correctness(dataset_task: dict, exe_path: str, work_dir: str):
    """
       correctness & functionality      gen.py -> exe -> compare.py

    Returns:
        (correctness: bool, functionality: bool)
    """
    data_dir = os.path.join(work_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1) gen.py
    ok, error = run_script_as_function(dataset_task.get("gen.py", ""), work_dir=work_dir)
    if not ok:
        print(f"  [GEN_FAILED] {str(error)[:160]}")
        return False, False

    # 2)
    if not run_exe(exe_path, work_dir):
        print("  [RUN_FAILED]")
        return False, False

    # 3) compare.py
    ok, compare_out = run_script_as_function(dataset_task.get("compare.py", ""), work_dir=work_dir)
    if not ok:
        print("  [COMPARE_FAILED]")
        return True, False

    # compare.py        "F"
    if "F" in compare_out:
        print("  [OUTPUT_MISMATCH]")
        return True, False

    return True, True


# ============================================================
#      NCU
# ============================================================
def eval_eff_only(executable_path: str, csv_output_path: str, gpu_id: int = 2):
    """        (BU, CE, Score)   None"""
    metric = [
        "dram__bytes.sum.peak_sustained",
        "dram__cycles_elapsed.avg.per_second",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
        "sm__cycles_elapsed.avg.per_second",
        "dram__bytes.sum.per_second",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed",
        "smsp__cycles_elapsed.avg.per_second",
        "sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    ]
    metrics = ",".join(metric)

    work_dir = os.path.dirname(executable_path)
    exe_name = os.path.basename(executable_path)

    ncu_command = ["ncu", "--metrics", metrics, "--csv", f"./{exe_name}"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    try:
        with open(csv_output_path, "w") as fcsv:
            result = subprocess.run(
                ncu_command,
                env=env,
                cwd=work_dir,
                stdout=fcsv,
                stderr=subprocess.PIPE,
                timeout=600,
                check=False,
                text=True,
            )

        if result.returncode != 0:
            print(" [NCU_ERROR]")
            if result.stderr.strip():
                print("---- ncu stderr ----")
                print(result.stderr[-4000:])
            return None

    except subprocess.TimeoutExpired:
        print(" [NCU_TIMEOUT]")
        return None
    except Exception as e:
        print(f" [NCU_EXCEPTION] {e}")
        return None

    #    CSV
    try:
        with open(csv_output_path, "r") as f:
            cnt = 0
            while True:
                ln = f.readline()
                if not ln:
                    break
                cnt += 1
                if "Host Name" in ln:
                    break

        df = pd.read_csv(csv_output_path, skiprows=cnt - 1)

        if df.empty:
            print(" [CSV_EMPTY]")
            return None
        if "Metric Value" not in df.columns or "Metric Name" not in df.columns:
            print(" [CSV_BAD_FORMAT]")
            return None

        df["Metric Value"] = df["Metric Value"].replace({",": ""}, regex=True).astype(float)
        dft = df.groupby(["Kernel Name", "Metric Name"]).sum()
        dfmetric = pd.pivot_table(dft, index="Kernel Name", columns="Metric Name", values="Metric Value")
        dfmetric["Count"] = df.groupby(["Kernel Name"]).count()["ID"].div(dfmetric.shape[1])

        required_cols = [
            "dram__bytes.sum.peak_sustained",
            "dram__cycles_elapsed.avg.per_second",
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
            "sm__cycles_elapsed.avg.per_second",
            "dram__bytes.sum.per_second",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",
            "smsp__cycles_elapsed.avg.per_second",
        ]
        missing = [c for c in required_cols if c not in dfmetric.columns]
        if missing:
            print(f" [MISSING_METRICS] {missing}")
            return None

        #
        dfmetric["Peak S FLOPs"] = (
            2 * dfmetric["sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"] / dfmetric["Count"]
        )
        dfmetric["PeakWork"] = dfmetric["Peak S FLOPs"] * dfmetric["sm__cycles_elapsed.avg.per_second"] / dfmetric["Count"]
        dfmetric["PeakTraffic"] = (
            dfmetric["dram__bytes.sum.peak_sustained"].div(dfmetric["Count"])
            * dfmetric["dram__cycles_elapsed.avg.per_second"].div(dfmetric["Count"])
        )

        dfmetric["S FLOPs"] = (
            2 * dfmetric["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"]
            + dfmetric["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"]
            + dfmetric["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"]
        )
        dfmetric["all FLOPs"] = dfmetric["S FLOPs"] / dfmetric["Count"]
        dfmetric["FLOP/s"] = dfmetric["all FLOPs"] * dfmetric["smsp__cycles_elapsed.avg.per_second"].div(dfmetric["Count"])
        dfmetric["AI DRAM"] = (
            (dfmetric["all FLOPs"] * dfmetric["smsp__cycles_elapsed.avg.per_second"].div(dfmetric["Count"]))
            .div(dfmetric["dram__bytes.sum.per_second"].div(dfmetric["Count"]))
        )

        #   kernel
        if dfmetric.shape[0] > 1:
            if (dfmetric["FLOP/s"] > 0).any():
                chosen = dfmetric["FLOP/s"].idxmax()
            else:
                chosen = dfmetric["dram__bytes.sum.per_second"].idxmax()
            print(f" [MULTI_KERNEL] pick: {chosen}")
            dfmetric = dfmetric.loc[[chosen]]

        flops = dfmetric["FLOP/s"].item()
        peak_work = dfmetric["PeakWork"].item()
        ai_dram = dfmetric["AI DRAM"].item()
        peak_traffic = dfmetric["PeakTraffic"].item()

        bandwidth_utilization = (
            dfmetric["dram__bytes.sum.per_second"].div(dfmetric["Count"]).item() / dfmetric["PeakTraffic"].item()
        )

        EPSILON = 1e-9
        if abs(flops) < EPSILON:
            compute_efficiency = 0.0
            score = bandwidth_utilization
        else:
            compute_efficiency = flops / peak_work
            roofline_limit = min(peak_work, ai_dram * peak_traffic)
            score = flops / roofline_limit

        return bandwidth_utilization, compute_efficiency, score

    except Exception as e:
        print(f" [CSV_PARSE_ERROR] {e}")
        return None


# ============================================================
#     validity +
# ============================================================
def get_code_validity(
    code: str,
    dataset_task: dict,
    work_dir: str,
    input_correctness: bool,
    input_functionality: bool,
    revalidate: bool,
    gen_py_code: str | None = None,
):
    """


    Returns:
        (correctness: bool, functionality: bool, executable_path: str|None)
    """
    if not revalidate:
        # trust    input correctness/functionality
        if input_correctness and input_functionality:
            #     gen.py
            if gen_py_code:
                ok, out = run_script_as_function(gen_py_code, work_dir=work_dir)
                if not ok:
                    return False, False, None

            executable_path = compile_code(code, work_dir)
            if executable_path is None:
                return False, False, None
            return input_correctness, input_functionality, executable_path
        else:
            return input_correctness, input_functionality, None

    # revalidate    +    + compare
    executable_path = compile_code(code, work_dir)
    if executable_path is None:
        return False, False, None

    correctness, functionality = evaluate_correctness(dataset_task, executable_path, work_dir)
    return correctness, functionality, executable_path


def write_zero_metrics(output_item: dict):
    """   0          """
    output_item["bandwidth_utilization"] = 0.0
    output_item["compute_efficiency"] = 0.0
    output_item["score"] = 0.0
    return output_item


def safe_rmtree(path: str):
    """best-effort     """
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
