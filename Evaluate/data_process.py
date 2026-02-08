#!/usr/bin/env python3
"""
data_process.py - Simplified statistics calculator for the external runner.

Requirements implemented:
- Remove all geometric mean calculations.
- Remove "Arith(No 0)" column (no "exclude zeros" stats).
- Only keep arithmetic mean including zeros: "Arith(Inc 0)".
- Do NOT write any files. This module only reads evalresult JSONL(s) and returns/prints strings.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


# ============================================================
# Best-version extraction (kept same logic as original)
# ============================================================
def extract_best_metrics_format1(entry: Dict[str, Any]) -> Tuple[bool, bool, float, float, float]:
    """
    Format1: multiple versions (flat): correctness1/2/3, functionality1/2/3,
    bandwidth_utilization1..., compute_efficiency1..., score1...

    Selection priority (unchanged):
      1) correctness=True AND functionality=True -> highest score
      2) correctness=True -> highest score
      3) functionality=True -> highest score
      4) else choose highest score

    Returns:
      (correctness, functionality, bandwidth, compute, score)
    """
    versions = []
    code_num = 1

    while f"correctness{code_num}" in entry or f"functionality{code_num}" in entry:
        correctness = bool(entry.get(f"correctness{code_num}", False))
        functionality = bool(entry.get(f"functionality{code_num}", False))
        bandwidth = float(entry.get(f"bandwidth_utilization{code_num}", 0.0) or 0.0)
        compute = float(entry.get(f"compute_efficiency{code_num}", 0.0) or 0.0)
        score = float(entry.get(f"score{code_num}", 0.0) or 0.0)

        versions.append(
            {
                "correctness": correctness,
                "functionality": functionality,
                "bandwidth": bandwidth,
                "compute": compute,
                "score": score,
            }
        )
        code_num += 1

    if not versions:
        return False, False, 0.0, 0.0, 0.0

    both = [v for v in versions if v["correctness"] and v["functionality"]]
    if both:
        best = max(both, key=lambda x: x["score"])
        return best["correctness"], best["functionality"], best["bandwidth"], best["compute"], best["score"]

    correct_only = [v for v in versions if v["correctness"]]
    if correct_only:
        best = max(correct_only, key=lambda x: x["score"])
        return best["correctness"], best["functionality"], best["bandwidth"], best["compute"], best["score"]

    functional_only = [v for v in versions if v["functionality"]]
    if functional_only:
        best = max(functional_only, key=lambda x: x["score"])
        return best["correctness"], best["functionality"], best["bandwidth"], best["compute"], best["score"]

    best = max(versions, key=lambda x: x["score"])
    return best["correctness"], best["functionality"], best["bandwidth"], best["compute"], best["score"]


def extract_best_metrics_format2(entry: Dict[str, Any]) -> Tuple[bool, bool, float, float, float]:
    """
    Format2: single code (possibly nested efficiency dict).
    Returns: (correctness, functionality, bandwidth, compute, score)
    """
    correctness = bool(entry.get("correctness", False))
    functionality = bool(entry.get("functionality", False))

    if "efficiency" in entry and isinstance(entry["efficiency"], dict):
        bandwidth = float(entry["efficiency"].get("bandwidth_utilization", 0.0) or 0.0)
        compute = float(entry["efficiency"].get("compute_efficiency", 0.0) or 0.0)
        score = float(entry["efficiency"].get("score", 0.0) or 0.0)
    else:
        bandwidth = float(entry.get("bandwidth_utilization", 0.0) or 0.0)
        compute = float(entry.get("compute_efficiency", 0.0) or 0.0)
        score = float(entry.get("score", 0.0) or 0.0)

    return correctness, functionality, bandwidth, compute, score


def detect_format(entry: Dict[str, Any]) -> str:
    """Detect entry format (format1: multi-version, format2: single)."""
    if "code1" in entry or "correctness1" in entry:
        return "format1"
    return "format2"


# ============================================================
# Metrics: arithmetic mean including zeros (only)
# ============================================================
def arithmetic_mean_including_zeros(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


# ============================================================
# Public API: compute stats from evalresult JSONL files
# ============================================================
def compute_stats_from_evalresult_files(jsonl_paths: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Read one or more evalresult JSONL files and compute stats for:
      - Pass@1 (use version1 for format1; single entry for format2)
      - Pass@3 (best-version selection for format1; single entry for format2)

    Returns:
      (stats_pass1, stats_pass3)
      Each stats dict includes:
        total, pass_correctness, pass_functionality,
        bandwidth_mean, compute_mean, score_mean
    """
    entries: List[Dict[str, Any]] = []

    for p in jsonl_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    total = len(entries)
    if total == 0:
        zero = {
            "total": 0,
            "pass_correctness": 0,
            "pass_functionality": 0,
            "bandwidth_mean": 0.0,
            "compute_mean": 0.0,
            "score_mean": 0.0,
        }
        return zero, zero

    # Pass@1 arrays
    p1_bw: List[float] = []
    p1_comp: List[float] = []
    p1_score: List[float] = []
    p1_corr = 0
    p1_func = 0

    # Pass@3 arrays (best version)
    p3_bw: List[float] = []
    p3_comp: List[float] = []
    p3_score: List[float] = []
    p3_corr = 0
    p3_func = 0

    for e in entries:
        fmt = detect_format(e)

        # PASS@1
        if fmt == "format1":
            c1 = bool(e.get("correctness1", False))
            f1 = bool(e.get("functionality1", False))
            bw1 = float(e.get("bandwidth_utilization1", 0.0) or 0.0)
            comp1 = float(e.get("compute_efficiency1", 0.0) or 0.0)
            sc1 = float(e.get("score1", 0.0) or 0.0)

            p1_corr += 1 if c1 else 0
            p1_func += 1 if f1 else 0
            p1_bw.append(bw1)
            p1_comp.append(comp1)
            p1_score.append(sc1)
        else:
            c, f, bw, comp, sc = extract_best_metrics_format2(e)
            p1_corr += 1 if c else 0
            p1_func += 1 if f else 0
            p1_bw.append(bw)
            p1_comp.append(comp)
            p1_score.append(sc)

        # PASS@3 (best version)
        if fmt == "format1":
            c, f, bw, comp, sc = extract_best_metrics_format1(e)
        else:
            c, f, bw, comp, sc = extract_best_metrics_format2(e)

        p3_corr += 1 if c else 0
        p3_func += 1 if f else 0
        p3_bw.append(bw)
        p3_comp.append(comp)
        p3_score.append(sc)

    stats1 = {
        "total": total,
        "pass_correctness": p1_corr,
        "pass_functionality": p1_func,
        "bandwidth_mean": arithmetic_mean_including_zeros(p1_bw),
        "compute_mean": arithmetic_mean_including_zeros(p1_comp),
        "score_mean": arithmetic_mean_including_zeros(p1_score),
    }

    stats3 = {
        "total": total,
        "pass_correctness": p3_corr,
        "pass_functionality": p3_func,
        "bandwidth_mean": arithmetic_mean_including_zeros(p3_bw),
        "compute_mean": arithmetic_mean_including_zeros(p3_comp),
        "score_mean": arithmetic_mean_including_zeros(p3_score),
    }

    return stats1, stats3


# ============================================================
# Table formatting: match your desired shape (only Arith(Inc 0))
# ============================================================
def format_stats_table(stats: Dict[str, Any], title: str = "BENCHMARK REPORT") -> str:
    """
    Format output to match the desired table shape:

    BENCHMARK REPORT | Total Samples: N
    Pass@X (Correctness):   a/N (xx.xx%)
    Pass@X (Functionality): b/N (yy.yy%)
    Metric | Arith(Inc 0)
    ...
    """
    total = int(stats.get("total", 0))
    pc = int(stats.get("pass_correctness", 0))
    pf = int(stats.get("pass_functionality", 0))

    bw = float(stats.get("bandwidth_mean", 0.0))
    comp = float(stats.get("compute_mean", 0.0))
    score = float(stats.get("score_mean", 0.0))

    pc_pct = (pc / total * 100.0) if total > 0 else 0.0
    pf_pct = (pf / total * 100.0) if total > 0 else 0.0

    sep_top = "=" * 100
    sep_mid = "-" * 100

    lines: List[str] = []
    lines.append(sep_top)
    lines.append(f"{title} | Total Samples: {total}")
    lines.append(sep_top)
    lines.append(f"Pass@ (Correctness):   {pc}/{total} ({pc_pct:.2f}%)")
    lines.append(f"Pass@ (Functionality): {pf}/{total} ({pf_pct:.2f}%)")
    lines.append(sep_mid)

    # Header: ONLY keep Arith(Inc 0)
    lines.append(f"{'Metric':<27} | {'Arith(Inc 0)':<14}")
    lines.append(sep_mid)

    # Values are printed as percentages (x100) to match original style
    lines.append(f"{'Bandwidth_Utilization':<27} | {bw*100:11.4f}%")
    lines.append(f"{'Compute_Efficiency':<27} | {comp*100:11.4f}%")
    lines.append(f"{'Score':<27} | {score*100:11.4f}%")

    lines.append(sep_top)
    return "\n".join(lines)


def format_two_tables(stats_pass1: Dict[str, Any], stats_pass3: Dict[str, Any]) -> Tuple[str, str]:
    """
    Convenience: return (pass1_table_str, pass3_table_str).
    Manager should print them (and nothing else).
    """
    t1 = format_stats_table(stats_pass1, title="BENCHMARK REPORT (PASS@1)")
    t3 = format_stats_table(stats_pass3, title="BENCHMARK REPORT (PASS@3)")
    return t1, t3
