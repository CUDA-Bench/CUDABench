#!/usr/bin/env python3
"""
manager.py - External slim evaluation runner (single-file mode).

Usage:
  python scripts/manager.py full <results_file.jsonl> [--gpu-id N] [--trust] [--dataset PATH]

Rules:
- Default GPU ID is 0.
- Default mode is revalidate (compile + run + compare). Use --trust to skip revalidation.
- Zero side effects: all intermediate files are created under a temporary directory and deleted on exit.
- Terminal output is limited to:
    * a single-line progress indicator
    * two final tables: Pass@1 and Pass@3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure scripts/ directory is importable when running as a script
THIS_DIR = Path(__file__).parent.resolve()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_from_json
import data_process


def _progress_line(done: int, total: int, width: int = 40) -> str:
    if total <= 0:
        total = 1
    pct = done / total
    filled = int(pct * width)
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    return f"{bar} {done}/{total} ({pct*100:6.2f}%)"


def parse_args():
    p = argparse.ArgumentParser(description="External slim evaluation runner (single-file mode).")
    p.add_argument("command", choices=["full"])
    p.add_argument("input", help="Path to a results JSON/JSONL file.")
    p.add_argument("--gpu-id", type=int, default=0, help="CUDA_VISIBLE_DEVICES id (default: 0)")
    p.add_argument("--trust", action="store_true", help="Trust input correctness/functionality (skip revalidate).")
    p.add_argument("--dataset", default=None, help="Optional dataset jsonl path. If omitted, auto-discovery is used.")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print("Input file not found.")
        sys.exit(2)

    # Use a temp directory to ensure zero side effects.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        # Mirror a minimal layout inside temp.
        results_dir = tmp_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        local_input = results_dir / input_path.name
        shutil.copy2(input_path, local_input)

        output_dir = tmp_root / "eval_out"
        temp_dir = tmp_root / "temp_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset tasks (needed for revalidate).
        dataset_tasks = {}
        dataset_path = None
        if args.dataset:
            dataset_path = str(Path(args.dataset).resolve())
        else:
            found = eval_from_json.find_dataset_file()
            if found:
                dataset_path = found

        if dataset_path:
            try:
                dataset_tasks = eval_from_json.load_dataset_tasks(dataset_path)
            except Exception:
                dataset_tasks = {}

        # Configure GPU id in the eval module.
        eval_from_json.GPU_ID = int(args.gpu_id)

        # Progress callback: write to the real stdout to bypass redirected stdout.
        real_stdout = sys.stdout

        def progress_callback(done_count: int, total_count: int):
            real_stdout.write("\r" + _progress_line(done_count, total_count) + " ")
            real_stdout.flush()

        # Redirect all stdout/stderr produced by eval_from_json to keep terminal clean.
        # progress_callback uses real_stdout, so it is still visible.
        revalidate = not bool(args.trust)

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                eval_from_json.process_json_file(
                    input_file=str(local_input),
                    output_dir=str(output_dir),
                    temp_dir=str(temp_dir),
                    dataset_tasks=dataset_tasks,
                    mode="pass3",
                    revalidate=revalidate,
                    silent=True,
                    progress_callback=progress_callback,
                )
        except Exception:
            real_stdout.write("\n")
            real_stdout.flush()
            print("Evaluation failed.")
            sys.exit(3)

        real_stdout.write("\n")
        real_stdout.flush()

        # Locate produced eval JSONL files under output_dir
        eval_files = list(output_dir.glob("*.jsonl"))
        if not eval_files:
            print("No eval output produced.")
            sys.exit(4)

        stats1, stats3 = data_process.compute_stats_from_evalresult_files([str(p) for p in eval_files])

        table1 = data_process.format_stats_table(stats1, "PASS@1 SUMMARY")
        table3 = data_process.format_stats_table(stats3, "PASS@3 (BEST VERSION) SUMMARY")
        print(table1)
        print(table3)


if __name__ == "__main__":
    main()
