#!/usr/bin/env python3
"""
manager.py - User-friendly evaluation runner.

Usage:
  python Evaluate/manager.py <filename.jsonl> [--gpu-id N] [--trust] [--dataset PATH]

Example:
  python Evaluate/manager.py qwen3-max-level1_pass3.jsonl
  python Evaluate/manager.py qwen3-max-level1_pass3.jsonl --gpu-id 2 --trust

Features:
- Automatically searches for JSONL files in Results/ directory
- Uses timestamped temp directories under Evaluate/temp/
- Cleans up temp directory before and after each run
- Default GPU ID is 0
- Default mode is revalidate (compile + run + compare)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure Evaluate/ directory is importable when running as a script
THIS_DIR = Path(__file__).parent.resolve()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_from_json
import data_process


def _progress_line(done: int, total: int, width: int = 40) -> str:
    """Generate a progress bar string."""
    if total <= 0:
        total = 1
    pct = done / total
    filled = int(pct * width)
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    return f"{bar} {done}/{total} ({pct*100:6.2f}%)"


def find_jsonl_in_results(filename: str, results_root: Path) -> Path:
    """
    Recursively search for a JSONL file in Results/ directory.
    
    Args:
        filename: The JSONL filename to search for
        results_root: Root Results/ directory
    
    Returns:
        Path to the found file
    
    Raises:
        SystemExit: If file not found or multiple matches found
    """
    if not results_root.exists():
        print(f"Error: Results directory not found: {results_root}")
        sys.exit(1)
    
    matches = list(results_root.rglob(filename))
    
    if len(matches) == 0:
        print(f"Error: File '{filename}' not found in {results_root}/")
        print(f"Searched recursively in all subdirectories.")
        sys.exit(1)
    
    if len(matches) > 1:
        print(f"Error: Multiple files named '{filename}' found:")
        for match in matches:
            print(f"  - {match.relative_to(results_root.parent)}")
        print("Please rename files to avoid conflicts or specify full path.")
        sys.exit(1)
    
    return matches[0]


def get_project_root() -> Path:
    """
    Find project root by looking for Datasets/ or Results/ directories.
    Assumes manager.py is in Evaluate/ directory.
    """
    current = THIS_DIR
    # Go up one level from Evaluate/
    project_root = current.parent
    
    # Verify we're in the right place
    if not (project_root / "Results").exists():
        print(f"Warning: Results/ directory not found at {project_root}")
        print(f"Creating Results/ directory...")
        (project_root / "Results").mkdir(exist_ok=True)
    
    return project_root


def clean_temp_directory(temp_root: Path):
    """Clean temp directory contents but keep the directory itself."""
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="CUDA Benchmark Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Evaluate/manager.py qwen3-max-level1_pass3.jsonl
  python Evaluate/manager.py qwen3-max-level1_pass3.jsonl --gpu-id 2
  python Evaluate/manager.py qwen3-max-level1_pass3.jsonl --trust --dataset Datasets/custom.jsonl
        """
    )
    p.add_argument("input", help="JSONL filename (will be searched in Results/ directory)")
    p.add_argument("--gpu-id", type=int, default=0, help="CUDA_VISIBLE_DEVICES id (default: 0)")
    p.add_argument("--trust", action="store_true", help="Trust input correctness/functionality (skip revalidation)")
    p.add_argument("--dataset", default=None, help="Dataset JSONL path (default: auto-discover in Datasets/)")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Determine project root
    project_root = get_project_root()
    results_dir = project_root / "Results"
    
    # Find input JSONL file (silently)
    input_path = find_jsonl_in_results(args.input, results_dir)
    
    # Setup timestamped temp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_root = THIS_DIR / "temp" / f"run_{timestamp}"
    
    # Clean before run (silently)
    clean_temp_directory(THIS_DIR / "temp")
    
    # Create timestamped run directory
    temp_root.mkdir(parents=True, exist_ok=True)
    
    output_dir = temp_root / "eval_out"
    temp_dir = temp_root / "temp_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset tasks (silently)
    dataset_tasks = {}
    dataset_path = None
    
    if args.dataset:
        dataset_path = str(Path(args.dataset).resolve())
    else:
        # Try to find dataset with new path structure
        found = eval_from_json.find_dataset_file()
        if found:
            dataset_path = found
    
    if dataset_path:
        try:
            dataset_tasks = eval_from_json.load_dataset_tasks(dataset_path)
        except Exception:
            dataset_tasks = {}
    
    # Configure GPU
    eval_from_json.GPU_ID = int(args.gpu_id)
    
    revalidate = not bool(args.trust)
    
    # Progress callback
    real_stdout = sys.stdout
    
    def progress_callback(done_count: int, total_count: int):
        real_stdout.write("\r" + _progress_line(done_count, total_count))
        real_stdout.flush()
    
    # Run evaluation with redirected output
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            eval_from_json.process_json_file(
                input_file=str(input_path),
                output_dir=str(output_dir),
                temp_dir=str(temp_dir),
                dataset_tasks=dataset_tasks,
                mode="pass3",
                revalidate=revalidate,
                silent=True,
                progress_callback=progress_callback,
            )
    except Exception as e:
        real_stdout.write("\n")
        real_stdout.flush()
        print(f"Evaluation failed: {e}")
        # Clean up after failure
        clean_temp_directory(THIS_DIR / "temp")
        sys.exit(3)
    
    real_stdout.write("\n")
    real_stdout.flush()
    print()
    
    # Compute statistics
    eval_files = list(output_dir.glob("*.jsonl"))
    if not eval_files:
        print("Error: No evaluation output produced.")
        clean_temp_directory(THIS_DIR / "temp")
        sys.exit(4)
    
    stats1, stats3 = data_process.compute_stats_from_evalresult_files([str(p) for p in eval_files])
    
    # Display results
    table1 = data_process.format_stats_table(stats1, "PASS@1 SUMMARY")
    table3 = data_process.format_stats_table(stats3, "PASS@3 (BEST VERSION) SUMMARY")
    print(table1)
    print()
    print(table3)
    
    # Clean up after successful run (silently)
    clean_temp_directory(THIS_DIR / "temp")


if __name__ == "__main__":
    main()