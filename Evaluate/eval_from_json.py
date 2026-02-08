#!/usr/bin/env python3
"""
eval_from_json.py - GPU code evaluation tool (performance evaluation).

Minimal adaptations for the external runner:
- Default GPU_ID is 0.
- process_json_file accepts silent/progress_callback; progress_callback is used for a progress bar.
- Non-essential console output can be suppressed via silent=True.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import shutil
import pandas as pd
import argparse
import contextlib
import io

#
START_INDEX = 0
DATASET = "100tasks_v3_prompts.jsonl"
GPU_ID = 0

def parse_args():
    """       """
    parser = argparse.ArgumentParser(
        description='GPU        ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  :
  #         input jsonl    correctness/functionality
  python eval_from_json.py results evalresult
  
  #
  python eval_from_json.py results evalresult --revalidate
  
  #
  python eval_from_json.py results evalresult temp_eval dataset.jsonl pass3 --revalidate
        """
    )
    
    #
    parser.add_argument('results_dir', help='           ')
    parser.add_argument('output_dir', help='        ')
    parser.add_argument('temp_dir', nargs='?', default='./temp_eval', help='         : ./temp_eval ')
    parser.add_argument('dataset_jsonl', nargs='?', default=None, help='          :      ')
    parser.add_argument('mode', nargs='?', default='pass3', help='    : pass1/pass3   : pass3 ')
    
    #
    parser.add_argument(
        '--revalidate',
        action='store_true',
        help='     correctness/functionality                 :    input jsonl    '
    )
    
    return parser.parse_args()


def remove_directory(path):
    """          """
    if not os.path.exists(path):
        return
    
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            remove_directory(item_path)
    
    os.rmdir(path)

def load_dataset_tasks(jsonl_path):
    """  100tasks_v3_prompts.jsonl       """
    tasks = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                tasks[task['id']] = task
    return tasks

# ============================================================
#    Working Directory Context & Run Script
# ============================================================
@contextlib.contextmanager
def working_directory(path):
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
            Python
    
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
#      NCU
# ============================================================
def eval_eff_only(executable_path, csv_output_path, gpu_id=GPU_ID):
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
        "sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"
    ]
    metrics = ",".join(metric)

    work_dir = os.path.dirname(executable_path)
    exe_name = os.path.basename(executable_path)

    ncu_command = ['ncu', '--metrics', metrics, '--csv', f'./{exe_name}']

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
                text=True
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
        with open(csv_output_path, 'r') as f:
            cnt = 0
            while True:
                ln = f.readline()
                if not ln:
                    break
                cnt += 1
                if 'Host Name' in ln:
                    break

        df = pd.read_csv(csv_output_path, skiprows=cnt-1)

        if df.empty:
            print(" [CSV_EMPTY]")
            return None
        if 'Metric Value' not in df.columns or 'Metric Name' not in df.columns:
            print(" [CSV_BAD_FORMAT]")
            return None

        df['Metric Value'] = df['Metric Value'].replace({',': ''}, regex=True).astype(float)
        dft = df.groupby(['Kernel Name', 'Metric Name']).sum()
        dfmetric = pd.pivot_table(dft, index='Kernel Name', columns='Metric Name', values='Metric Value')
        dfmetric['Count'] = df.groupby(['Kernel Name']).count()['ID'].div(dfmetric.shape[1])

        required_cols = [
            'dram__bytes.sum.peak_sustained',
            'dram__cycles_elapsed.avg.per_second',
            'sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained',
            'sm__cycles_elapsed.avg.per_second',
            'dram__bytes.sum.per_second',
            'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed',
            'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed',
            'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed',
            'smsp__cycles_elapsed.avg.per_second'
        ]
        missing = [c for c in required_cols if c not in dfmetric.columns]
        if missing:
            print(f" [MISSING_METRICS] {missing}")
            return None

        #
        dfmetric['Peak S FLOPs'] = 2 * dfmetric['sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained'] / dfmetric['Count']
        dfmetric['PeakWork'] = dfmetric['Peak S FLOPs'] * dfmetric['sm__cycles_elapsed.avg.per_second'] / dfmetric['Count']
        dfmetric['PeakTraffic'] = dfmetric['dram__bytes.sum.peak_sustained'].div(dfmetric['Count']) * \
                                  dfmetric['dram__cycles_elapsed.avg.per_second'].div(dfmetric['Count'])

        dfmetric['S FLOPs'] = 2 * dfmetric['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed'] + \
                              dfmetric['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'] + \
                              dfmetric['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed']
        dfmetric['all FLOPs'] = dfmetric['S FLOPs'] / dfmetric['Count']
        dfmetric['FLOP/s'] = dfmetric['all FLOPs'] * dfmetric['smsp__cycles_elapsed.avg.per_second'].div(dfmetric['Count'])
        dfmetric['AI DRAM'] = (dfmetric['all FLOPs'] * dfmetric['smsp__cycles_elapsed.avg.per_second'].div(dfmetric['Count'])).div(
                              dfmetric['dram__bytes.sum.per_second'].div(dfmetric['Count']))

        #   kernel
        if dfmetric.shape[0] > 1:
            if (dfmetric['FLOP/s'] > 0).any():
                chosen = dfmetric['FLOP/s'].idxmax()
            else:
                chosen = dfmetric['dram__bytes.sum.per_second'].idxmax()
            print(f" [MULTI_KERNEL] pick: {chosen}")
            dfmetric = dfmetric.loc[[chosen]]

        flops = dfmetric['FLOP/s'].item()
        peak_work = dfmetric['PeakWork'].item()
        ai_dram = dfmetric['AI DRAM'].item()
        peak_traffic = dfmetric['PeakTraffic'].item()

        bandwidth_utilization = (
            dfmetric['dram__bytes.sum.per_second']
            .div(dfmetric['Count'])
            .item()
            / dfmetric['PeakTraffic'].item()
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
#
# ============================================================
def compile_code(code, work_dir, code_filename="kernel.cu"):
    """  CUDA  """
    cu_file_path = os.path.join(work_dir, code_filename)
    executable_path = os.path.join(work_dir, "kernel")

    with open(cu_file_path, 'w') as f:
        f.write(code)

    compile_command = ['nvcc', cu_file_path, '-o', executable_path]
    try:
        subprocess.run(compile_command, check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return executable_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

# ============================================================
#               rerun
# ============================================================
def run_exe(exe_path, work_dir, timeout=60):
    """

    
    Returns:
        bool:            0
    """
    try:
        exe_file = "./" + os.path.basename(exe_path)
        result = subprocess.run(
            [exe_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            cwd=work_dir
        )
        return result.returncode == 0
    except Exception:
        return False

# ============================================================
#             rerun
# ============================================================
def evaluate_correctness(dataset_task, exe_path, work_dir):
    """

    
    Returns:
        (correctness: bool, functionality: bool)
    """
    data_dir = os.path.join(work_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1.    gen.py
    ok, error = run_script_as_function(
        dataset_task["gen.py"],
        work_dir=work_dir
    )
    if not ok:
        print(f"  [GEN_FAILED] {str(error)[:100]}")
        return False, False

    # 2.    CUDA
    if not run_exe(exe_path, work_dir):
        print("  [RUN_FAILED]")
        return False, False

    # 3.    compare.py
    ok, compare_out = run_script_as_function(
        dataset_task["compare.py"],
        work_dir=work_dir
    )
    if not ok:
        print("  [COMPARE_FAILED]")
        return True, False  #

    if "F" in compare_out:
        print("  [OUTPUT_MISMATCH]")
        return True, False

    return True, True

# ============================================================
#       gen.py    trust
# ============================================================
def run_gen_py(gen_py_code, work_dir):
    """   gen.py       """
    gen_py_path = os.path.join(work_dir, 'gen.py')
    with open(gen_py_path, 'w') as f:
        f.write(gen_py_code)

    data_dir = os.path.join(work_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    try:
        result = subprocess.run(
            ['python', 'gen.py'],
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            text=True
        )
        
        if result.returncode != 0:
            print(f" [GENPY_ERROR] Return code: {result.returncode}")
            if result.stderr.strip():
                print("---- gen.py stderr ----")
                print(result.stderr[-4000:])
            if result.stdout.strip():
                print("---- gen.py stdout ----")
                print(result.stdout[-2000:])
            return False
        
        #
        if os.path.exists(data_dir):
            data_files = os.listdir(data_dir)
            if not data_files:
                print(" [GENPY_WARNING] gen.py ran successfully but no files in data/")
                print(f" [DEBUG] Work dir: {work_dir}")
                print(f" [DEBUG] Data dir exists: {os.path.exists(data_dir)}")
                return False
            else:
                return True
        else:
            print(" [GENPY_ERROR] data/ directory not found after gen.py execution")
            return False
            
    except subprocess.TimeoutExpired:
        print(" [GENPY_TIMEOUT] (180s)")
        return False
    except Exception as e:
        print(f" [GENPY_EXCEPTION] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
#
# ============================================================
def get_code_validity(
    code,
    dataset_task,
    work_dir,
    input_correctness,
    input_functionality,
    revalidate,
    gen_py_code=None
):
    """

    
    Args:
        code: CUDA
        dataset_task:            gen.py, compare.py
        work_dir:
        input_correctness: input jsonl    correctness
        input_functionality: input jsonl    functionality
        revalidate: True=    , False=   input jsonl
        gen_py_code: gen.py    trust
        
    Returns:
        (correctness: bool, functionality: bool, executable_path: str|None)
    """
    if not revalidate:
        #         input jsonl
        if input_correctness and input_functionality:
            #     gen.py
            if gen_py_code and not run_gen_py(gen_py_code, work_dir):
                return False, False, None
            
            #
            executable_path = compile_code(code, work_dir)
            if executable_path is None:
                return False, False, None
            return input_correctness, input_functionality, executable_path
        else:
            #
            return input_correctness, input_functionality, None
    
    else:
        #
        # 1.
        executable_path = compile_code(code, work_dir)
        if executable_path is None:
            return False, False, None
        
        # 2.
        correctness, functionality = evaluate_correctness(
            dataset_task, executable_path, work_dir
        )
        return correctness, functionality, executable_path

# ============================================================
#
# ============================================================
def extract_code_versions(item):
    """        """
    versions = []
    
    #      code
    if 'code' in item:
        versions.append({
            'code': item.get('code', ''),
            'correctness': item.get('correctness', False),
            'functionality': item.get('functionality', False),
            'version_suffix': ''
        })
    
    #     code1, code2...
    i = 1
    while f'code{i}' in item:
        versions.append({
            'code': item.get(f'code{i}', ''),
            'correctness': item.get(f'correctness{i}', False),
            'functionality': item.get(f'functionality{i}', False),
            'version_suffix': f'_v{i}'
        })
        i += 1
    
    return versions

def write_zero_metrics(output_item, version_suffix):
    """Write zero performance metrics for a given version."""
    if version_suffix == '':
        output_item['bandwidth_utilization'] = 0.0
        output_item['compute_efficiency'] = 0.0
        output_item['score'] = 0.0
    else:
        n = version_suffix.replace('_v', '')
        output_item[f'bandwidth_utilization{n}'] = 0.0
        output_item[f'compute_efficiency{n}'] = 0.0
        output_item[f'score{n}'] = 0.0
    return output_item

# ============================================================
#
# ============================================================
def process_json_file(input_file, output_dir, temp_dir, dataset_tasks, mode='pass3', revalidate=False, silent=False, progress_callback=None):
    """    JSON            """
    if not silent:
        print(f"\n{'='*60}")
    if not silent:
        print(f"Processing: {os.path.basename(input_file)}")
    if not silent:
        print(f"Mode: {mode.upper()} | Revalidate: {revalidate}")
    if not silent:
        print(f"{'='*60}")

    input_filename = Path(input_file).stem

    #
    input_temp_dir = os.path.join(temp_dir, input_filename)
    if os.path.exists(input_temp_dir):
        if not silent:
            print(f"    Cleaning existing temp directory...")
        shutil.rmtree(input_temp_dir, ignore_errors=True)

    #
    out_base = f"{input_filename}_eval"
    output_path_jsonl = os.path.join(output_dir, f"{out_base}.jsonl")

    if os.path.exists(output_path_jsonl):
        if not silent:
            print(f"    Removing existing output file...")
        os.remove(output_path_jsonl)

    if not silent:
        print(f"Starting fresh evaluation...\n")

    #         JSON    / JSONL
    with open(input_file, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = []
            for line in content.split('\n'):
                if line.strip():
                    data.append(json.loads(line))

    total_tasks = len(data)
    os.makedirs(output_dir, exist_ok=True)

    #
    tasks_written = 0
    versions_total = 0
    versions_ok = 0
    versions_zero_invalid = 0
    versions_zero_env = 0
    versions_zero_evalfail = 0

    #
    for idx, item in enumerate(data):
        if idx < START_INDEX:
            continue

        task_id = item.get('id')
        task_name = item.get('task_name', 'Unknown')

        code_versions = extract_code_versions(item)

        # Pass@1
        if mode == 'pass1' and code_versions:
            code_versions = code_versions[0:1]

        #              id   task_name
        output_item = {
            'id': item.get('id'),
            'task_name': item.get('task_name', 'Unknown')
        }

        #        code
        if not code_versions:
            if not silent:
                print(f"[{idx+1}/{total_tasks}] ID={str(task_id):>3} {task_name:25s} (no code fields) -> write as-is")
            with open(output_path_jsonl, "a") as fa:
                fa.write(json.dumps(output_item) + "\n")
            tasks_written += 1
            if progress_callback:
                progress_callback(tasks_written, total_tasks)
            continue

        versions_total += len(code_versions)

        # dataset / gen.py
        has_dataset = task_id in dataset_tasks
        dataset_task = dataset_tasks.get(task_id, {})
        gen_py_code = dataset_task.get('gen.py', '') if has_dataset else ''
        can_run_eval = has_dataset and bool(gen_py_code)

        task_print_buffer = []
        task_print_buffer.append(
            f"[{idx+1}/{total_tasks}] ID={str(task_id):>3} {task_name:25s} ({len(code_versions)} version(s))"
        )

        for version_idx, version_info in enumerate(code_versions):
            version_suffix = version_info['version_suffix']
            code = version_info.get('code', '')
            input_correctness = version_info.get('correctness', False)
            input_functionality = version_info.get('functionality', False)

            version_line = f"  [{version_idx+1}/{len(code_versions)}] Testing{version_suffix}... "

            #         input   correctness/functionality
            if not revalidate:
                ok = (input_correctness is True and input_functionality is True)
                
                #
                if (not ok) or (not code):
                    output_item = write_zero_metrics(output_item, version_suffix)
                    versions_zero_invalid += 1
                    reason = "not correct/functional" if not ok else "empty code"
                    task_print_buffer.append(version_line + f"SKIP ({reason}) -> metrics=0")
                    continue

                #
                if not can_run_eval:
                    output_item = write_zero_metrics(output_item, version_suffix)
                    versions_zero_env += 1
                    if not has_dataset:
                        task_print_buffer.append(version_line + "SKIP (no dataset) -> metrics=0")
                    else:
                        task_print_buffer.append(version_line + "SKIP (no gen.py) -> metrics=0")
                    continue
            
            #
            else:
                if not code:
                    output_item = write_zero_metrics(output_item, version_suffix)
                    versions_zero_invalid += 1
                    task_print_buffer.append(version_line + "SKIP (empty code) -> metrics=0")
                    continue
                
                if not can_run_eval:
                    output_item = write_zero_metrics(output_item, version_suffix)
                    versions_zero_env += 1
                    if not has_dataset:
                        task_print_buffer.append(version_line + "SKIP (no dataset) -> metrics=0")
                    else:
                        task_print_buffer.append(version_line + "SKIP (no gen.py) -> metrics=0")
                    continue

            #
            work_dir = os.path.join(temp_dir, input_filename, f"task_{task_id}{version_suffix}")
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            os.makedirs(work_dir, exist_ok=True)

            version_success = False
            try:
                #
                correctness, functionality, executable_path = get_code_validity(
                    code=code,
                    dataset_task=dataset_task,
                    work_dir=work_dir,
                    input_correctness=input_correctness,
                    input_functionality=input_functionality,
                    revalidate=revalidate,
                    gen_py_code=gen_py_code
                )
                
                #    correctness   functionality
                if version_suffix == '':
                    output_item['correctness'] = correctness
                    output_item['functionality'] = functionality
                else:
                    n = version_suffix.replace('_v', '')
                    output_item[f'correctness{n}'] = correctness
                    output_item[f'functionality{n}'] = functionality
                
                #            functionality=False
                if executable_path is None or not functionality:
                    output_item = write_zero_metrics(output_item, version_suffix)
                    versions_zero_evalfail += 1
                    if executable_path is None:
                        task_print_buffer.append(version_line + "FAIL (compile) -> metrics=0")
                    else:
                        task_print_buffer.append(version_line + "FAIL (functionality) -> metrics=0")
                    continue

                #
                csv_path = os.path.join(work_dir, "ncu.csv")
                eval_result = eval_eff_only(executable_path, csv_path)
                if eval_result is None:
                    output_item = write_zero_metrics(output_item, version_suffix)
                    versions_zero_evalfail += 1
                    task_print_buffer.append(version_line + "FAIL (eval) -> metrics=0")
                    continue

                bu, ce, score = eval_result

                #
                if version_suffix == '':
                    output_item['bandwidth_utilization'] = round(bu, 5)
                    output_item['compute_efficiency'] = round(ce, 5)
                    output_item['score'] = round(score, 5)
                else:
                    n = version_suffix.replace('_v', '')
                    output_item[f'bandwidth_utilization{n}'] = round(bu, 5)
                    output_item[f'compute_efficiency{n}'] = round(ce, 5)
                    output_item[f'score{n}'] = round(score, 5)

                versions_ok += 1
                version_success = True
                task_print_buffer.append(version_line + "OK")

            finally:
                data_path = os.path.join(work_dir, "data")
                if os.path.exists(data_path):
                    shutil.rmtree(data_path, ignore_errors=True)

                #
                if version_success:
                    shutil.rmtree(work_dir, ignore_errors=True)

        if not silent:
            print('\n'.join(task_print_buffer))

        #          task
        with open(output_path_jsonl, "a") as fa:
            fa.write(json.dumps(output_item) + "\n")
        tasks_written += 1
        if progress_callback:
            progress_callback(tasks_written, total_tasks)

    if not silent:
        print(f"\n{'='*60}")
    if not silent:
        print(f"  Results: {output_path_jsonl}")
    if not silent:
        print(f"  Tasks written: {tasks_written}/{total_tasks - START_INDEX}")
    if not silent:
        print(f"  Versions total: {versions_total}")
    if not silent:
        print(f"  Versions OK: {versions_ok}")
    if not silent:
        print(f"  Versions -> metrics=0 (invalid): {versions_zero_invalid}")
    if not silent:
        print(f"  Versions -> metrics=0 (no dataset/gen.py): {versions_zero_env}")
    if not silent:
        print(f"  Versions -> metrics=0 (eval failed): {versions_zero_evalfail}")
    if not silent:
        print(f"{'='*60}")


def batch_process(results_dir, output_dir, temp_dir, dataset_jsonl, mode='pass3', revalidate=False):
    """          JSON/JSONL   """
    print(f"\n{'#'*60}")
    print(f"#       ")
    print(f"{'#'*60}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Temp dir:    {temp_dir}")
    print(f"Dataset:     {dataset_jsonl}")
    print(f"Mode:        {mode}")
    print(f"Revalidate:  {revalidate}")
    print(f"{'#'*60}\n")
    
    #
    print("Loading dataset...")
    if not os.path.exists(dataset_jsonl):
        print(f"  Error: Dataset not found - {dataset_jsonl}")
        return False
    
    try:
        dataset_tasks = load_dataset_tasks(dataset_jsonl)
        print(f"  Loaded {len(dataset_tasks)} tasks\n")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return False
    
    #
    if not os.path.exists(results_dir):
        print(f"  Error: Results directory not found - {results_dir}")
        return False
    
    all_files = os.listdir(results_dir)
    json_files = [f for f in all_files if f.endswith('.json') or f.endswith('.jsonl')]
    
    if not json_files:
        print(f"  No JSON/JSONL files found in {results_dir}")
        return False
    
    print(f"Found {len(json_files)} file(s):")
    for f in json_files:
        print(f"  - {f}")
    
    #
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    #
    success_count = 0
    for idx, json_file in enumerate(json_files, 1):
        print(f"\n{'#'*60}")
        print(f"# Processing file {idx}/{len(json_files)}: {json_file}")
        print(f"{'#'*60}")
        
        input_path = os.path.join(results_dir, json_file)
        try:
            process_json_file(input_path, output_dir, temp_dir, dataset_tasks, mode, revalidate)
            success_count += 1
        except Exception as e:
            print(f"\n  Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nContinuing with next file...")
            continue
    
    print(f"\n{'#'*60}")
    print(f"# Batch processing complete")
    print(f"{'#'*60}")
    print(f"Total files:    {len(json_files)}")
    print(f"Success:        {success_count}")
    print(f"Failed:         {len(json_files) - success_count}")
    print(f"{'#'*60}\n")
    
    return success_count > 0

def find_dataset_file():
    """               """
    possible_locations = [
        DATASET,
        f'datasets/{DATASET}',
        f'./{DATASET}',
        f'./datasets/{DATASET}',
        f'../{DATASET}',
        f'../datasets/{DATASET}',
        f'../../{DATASET}',
        f'../../datasets/{DATASET}',
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            return os.path.abspath(loc)
    
    return None

def print_help():
    """      """
    help_text = """
GPU


  1.           results/
  2.    CUDA     NCU
  3.
  4.        evalresult/


          input jsonl    correctness/functionality
  --revalidate            correctness/functionality

  :
  python eval_from_json.py <results  > <evalresult  > [  ]

    :
  results
  evalresult
  temp           (  )           ./temp_eval
  dataset        (  )
  mode             (  ) pass1/pass3    pass3

    :
  --revalidate         correctness/functionality

  :
  #         input jsonl
  python eval_from_json.py results evalresult
  
  #
  python eval_from_json.py results evalresult --revalidate
  
  #
  python eval_from_json.py results evalresult temp_eval datasets/100tasks_v3_prompts.jsonl pass3 --revalidate

    :
  1.
     python eval_from_json.py results evalresult
  
  2.            GPU
     python eval_from_json.py results evalresult --revalidate

  :
  - CUDA toolkit (nvcc)
  - NVIDIA Nsight Compute (ncu)
  - pandas
"""
    print(help_text)

if __name__ == "__main__":
    args = parse_args()
    
    #
    if args.dataset_jsonl is None:
        print("  Searching for dataset file...")
        args.dataset_jsonl = find_dataset_file()
        if args.dataset_jsonl:
            print(f"  Found: {args.dataset_jsonl}\n")
        else:
            print(f"  Error: Could not find {DATASET}")
            print("\nSearched locations:")
            print(f"  - ./{DATASET}")
            print(f"  - ./datasets/{DATASET}")
            print(f"  - ../{DATASET}")
            print(f"  - ../../{DATASET}")
            print("\nPlease specify the path manually:")
            print(f"  python eval_from_json.py results evalresult temp_eval <path_to_dataset>")
            sys.exit(1)
    
    #
    success = batch_process(
        args.results_dir,
        args.output_dir,
        args.temp_dir,
        args.dataset_jsonl,
        args.mode,
        args.revalidate
    )
    sys.exit(0 if success else 1)
