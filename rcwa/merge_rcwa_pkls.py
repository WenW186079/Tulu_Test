# 只看报告（不保存合并文件）
python merge_rcwa_pkls.py runs/*.pkl --report-only

# 合并并另存为 pkl，同时导出 CSV
python merge_rcwa_pkls.py a.pkl b.pkl c.pkl --out-pkl merged.pkl --out-csv merged.csv


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple RCWA SimulationLibrary .pkl files produced by metabox.modeling.save_simulation_library.
- Prints a per-file summary of what's inside
- Checks whether core settings are consistent (wavelengths, harmonics, resolution, periodicity, etc.)
- Reports any differences
- Concatenates the samples (feature sweep) along the sample axis and saves a merged .pkl
- Optionally exports a merged CSV (t_power and r_power)

Usage examples:
  python merge_rcwa_pkls.py runs/*.pkl --out-pkl merged.pkl --out-csv merged.csv
  python merge_rcwa_pkls.py a.pkl b.pkl c.pkl --report-only

Requires: tensorflow, numpy, pandas, metabox
"""
import os
import sys
import glob
import pickle
import argparse
import copy
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from metabox import rcwa, modeling


def load_simlib(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def safe_get(obj, attr, default=None):
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def summarize_simlib(simlib) -> Dict[str, Any]:
    """Return a concise, comparable summary for a SimulationLibrary."""
    # Incidence
    inc = simlib.incidence
    wl = np.array(safe_get(inc, "wavelength", []))
    theta = tuple(safe_get(inc, "theta", ()))
    phi = tuple(safe_get(inc, "phi", ()))
    jones = tuple(safe_get(inc, "jones_vector", ()))

    # Sim config
    cfg = simlib.sim_config
    harmonics = tuple(safe_get(cfg, "xy_harmonics", ()))
    resolution = int(safe_get(cfg, "resolution", -1))
    minibatch = int(safe_get(cfg, "minibatch_size", -1))

    # Protocell / Unit cell
    proto = simlib.protocell
    uc = safe_get(proto, "unit_cell", None)
    periodicity = tuple(safe_get(uc, "periodicity", ()))
    refl_index = safe_get(uc, "refl_index", None)
    tran_index = safe_get(uc, "tran_index", None)

    # Derived sizes
    fv = np.array(simlib.feature_values)
    num_samples = int(fv.size if fv.ndim == 1 else fv.shape[-1])
    num_wavelengths = int(wl.size)

    return {
        "num_samples": num_samples,
        "num_wavelengths": num_wavelengths,
        "wavelength_min": float(wl.min()) if wl.size else None,
        "wavelength_max": float(wl.max()) if wl.size else None,
        "theta": theta,
        "phi": phi,
        "jones": jones,
        "harmonics": harmonics,
        "resolution": resolution,
        "minibatch": minibatch,
        "periodicity": periodicity,
        "refl_index": refl_index,
        "tran_index": tran_index,
    }


def compare_values(a, b, name, issues, rtol=1e-12, atol=1e-15):
    """Append human-readable diffs to 'issues' if a and b differ."""
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        a_arr = np.array(a)
        b_arr = np.array(b)
        if a_arr.shape != b_arr.shape or not np.allclose(a_arr, b_arr, rtol=rtol, atol=atol, equal_nan=True):
            issues.append(f"{name} differs: {a} vs {b}")
    else:
        if a != b:
            issues.append(f"{name} differs: {a} vs {b}")


def check_compatibility(summaries: List[Dict[str, Any]]) -> List[str]:
    """Return a list of issues across all summaries (pairwise vs the first)."""
    issues = []
    if not summaries:
        return ["No files loaded."]

    base = summaries[0]
    for i, s in enumerate(summaries[1:], start=1):
        prefix = f"[file #{i+1}] "
        compare_values(s["num_wavelengths"], base["num_wavelengths"], prefix + "num_wavelengths", issues)
        compare_values((s["wavelength_min"], s["wavelength_max"]),
                       (base["wavelength_min"], base["wavelength_max"]),
                       prefix + "wavelength_range", issues)
        compare_values(s["theta"], base["theta"], prefix + "theta", issues)
        compare_values(s["phi"], base["phi"], prefix + "phi", issues)
        compare_values(s["jones"], base["jones"], prefix + "jones_vector", issues)
        compare_values(s["harmonics"], base["harmonics"], prefix + "xy_harmonics", issues)
        compare_values(s["resolution"], base["resolution"], prefix + "resolution", issues)
        compare_values(s["periodicity"], base["periodicity"], prefix + "periodicity", issues)
        compare_values(s["refl_index"], base["refl_index"], prefix + "refl_index", issues)
        compare_values(s["tran_index"], base["tran_index"], prefix + "tran_index", issues)
    return issues


def concat_feature_values(feature_values_list: List[np.ndarray]) -> np.ndarray:
    arrs = []
    for fv in feature_values_list:
        a = np.array(fv)
        if a.ndim == 1:
            arrs.append(a)
        elif a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1):
            arrs.append(a.reshape(-1))
        else:
            # Generic last-dim concat
            arrs.append(a.reshape(-1))
    return np.concatenate(arrs, axis=0)


def concat_sim_results(sim_results: List[Any]) -> Any:
    """Concatenate SimResult-like objects along the 'sample' axis (axis=1)."""
    if len(sim_results) == 1:
        return sim_results[0]

    merged = copy.deepcopy(sim_results[0])
    tensor_attr_names = []
    # Discover tensor-like attributes on the first result
    for name in dir(sim_results[0]):
        if name.startswith("_"):
            continue
        try:
            val = getattr(sim_results[0], name)
        except Exception:
            continue
        # Heuristic: TF tensors have .shape and .numpy()
        if hasattr(val, "shape") and callable(getattr(val, "numpy", None)):
            tensor_attr_names.append(name)

    for name in tensor_attr_names:
        parts = []
        for res in sim_results:
            t = getattr(res, name)
            parts.append(t)
        try:
            merged_t = tf.concat(parts, axis=1)  # axis=1 is the 'sample' dimension per your script
        except Exception as e:
            raise RuntimeError(f"Failed to concat attribute '{name}': {e}")
        setattr(merged, name, merged_t)

    return merged


def export_csv(sim_lib, csv_path: str):
    """Export t_power/r_power (and basics) as a CSV."""
    sim_result = sim_lib.simulation_output
    t_power = getattr(sim_result, "t_power", None)
    r_power = getattr(sim_result, "r_power", None)

    if t_power is None or r_power is None:
        raise RuntimeError("simulation_output lacks t_power or r_power.")

    wavelengths = np.array(sim_lib.incidence.wavelength)
    feature_values = np.array(sim_lib.feature_values).squeeze()
    xy_harmonics = sim_lib.sim_config.xy_harmonics

    t_power = t_power.numpy().squeeze()
    r_power = r_power.numpy().squeeze()

    rows = []
    if wavelengths.size == 1 and np.ndim(feature_values) == 0:
        rows.append({
            "wavelength (m)": float(wavelengths),
            "pillar width (m)": float(feature_values),
            "t_power": float(t_power),
            "r_power": float(r_power),
            "x_harmonics": xy_harmonics[0],
            "y_harmonics": xy_harmonics[1],
        })
    elif wavelengths.size == 1 and np.ndim(feature_values) == 1:
        for j, width in enumerate(feature_values):
            rows.append({
                "wavelength (m)": float(wavelengths),
                "pillar width (m)": float(width),
                "t_power": float(t_power[j]),
                "r_power": float(r_power[j]),
                "x_harmonics": xy_harmonics[0],
                "y_harmonics": xy_harmonics[1],
            })
    elif wavelengths.size > 1 and np.ndim(feature_values) == 0:
        for i, wl in enumerate(wavelengths):
            rows.append({
                "wavelength (m)": float(wl),
                "pillar width (m)": float(feature_values),
                "t_power": float(t_power[i]),
                "r_power": float(r_power[i]),
                "x_harmonics": xy_harmonics[0],
                "y_harmonics": xy_harmonics[1],
            })
    else:
        for i, wl in enumerate(wavelengths):
            for j, width in enumerate(feature_values):
                rows.append({
                    "wavelength (m)": float(wl),
                    "pillar width (m)": float(width),
                    "t_power": float(t_power[i, j]),
                    "r_power": float(r_power[i, j]),
                    "x_harmonics": xy_harmonics[0],
                    "y_harmonics": xy_harmonics[1],
                })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Merge RCWA SimulationLibrary pickles, check consistency, and export.")
    parser.add_argument("inputs", nargs="+", help="Input .pkl files (or globs).")
    parser.add_argument("--out-pkl", default="merged_simlib.pkl", help="Path to save merged SimulationLibrary .pkl")
    parser.add_argument("--out-csv", default=None, help="Optional CSV path to export merged t_power/r_power")
    parser.add_argument("--report-only", action="store_true", help="Only print report, don't save merged outputs")
    args = parser.parse_args()

    # Expand globs
    paths = []
    for pat in args.inputs:
        expanded = sorted(glob.glob(pat))
        if not expanded and os.path.isfile(pat):
            expanded = [pat]
        paths.extend(expanded)

    if not paths:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    # Load
    libs = []
    for p in paths:
        try:
            lib = load_simlib(p)
            libs.append(lib)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}", file=sys.stderr)

    if not libs:
        print("No valid SimulationLibrary objects loaded.", file=sys.stderr)
        sys.exit(1)

    # Summaries
    print("\n=== File summaries ===")
    summaries = []
    for p, lib in zip(paths, libs):
        s = summarize_simlib(lib)
        summaries.append(s)
        print(f"* {p}")
        for k, v in s.items():
            print(f"  - {k}: {v}")

    # Consistency checks
    print("\n=== Consistency check (vs first file) ===")
    issues = check_compatibility(summaries)
    if issues:
        for line in issues:
            print("! " + line)
    else:
        print("All core settings are consistent.")

    # Proceed to merge even if issues exist, but warn the user.
    if issues:
        print("\n[WARNING] Differences detected. Merging will assume identical wavelength grid and physics;")
        print("          results are concatenated along the *sample* axis (feature sweep). Please review the issues above.")

    if args.report_only:
        return

    # Build merged SimulationLibrary
    base = libs[0]
    merged_feature_values = concat_feature_values([lib.feature_values for lib in libs])
    merged_sim_result = concat_sim_results([lib.simulation_output for lib in libs])

    merged_lib = modeling.SimulationLibrary(
        protocell=base.protocell,
        incidence=base.incidence,
        sim_config=base.sim_config,
        feature_values=np.array(merged_feature_values),
        simulation_output=merged_sim_result
    )

    # Save merged pkl
    out_pkl = args.out_pkl
    os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(merged_lib, f)
    print(f"\nSaved merged SimulationLibrary to: {out_pkl}")

    # Optional CSV
    if args.out_csv:
        csv_path = args.out_csv
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        export_csv(merged_lib, csv_path)
        print(f"Saved merged CSV to: {csv_path}")


if __name__ == "__main__":
    main()
