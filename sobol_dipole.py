#!/usr/bin/env python
"""
Sobol testing of GAP dipole-fitting descriptor hyperparameters.

Orchestration layer: generates a SALib Sobol design over the SWEEP hyperparameters, runs each
sample through the single-config pipeline in gap_dipole.evaluate() (gap_fit -> quip -> trim ->
errorcalc), then reports:
  * variance-based sensitivity indices (S1 / ST) -> which hyperparameters drive RMSE
  * the sampled hyperparameter configuration with the lowest RMSE

The fitting mechanics and all shared configuration live in gap_dipole.py (one source of truth).
To fit/score a SINGLE configuration, use that module directly:  python gap_dipole.py --help

Run under the quipenv interpreter (needs ase, numpy, SALib):
    python sobol_dipole.py            # full pilot
    python sobol_dipole.py --smoke    # single nominal fit only (validates the pipeline)
    python sobol_dipole.py --n 32     # larger design (runs = n*(D+2))
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time

import numpy as np
from ase.io import read, write
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

import gap_dipole as core
from gap_dipole import (HYPERPARAMS, PER_SPECIES_CAPABLE, SPECIES_Z, DESCRIPTOR, PRED_KEY,
                        RUNS_DIR, BASE_DIR, TRAIN_FULL, TEST_FULL,
                        species_key, build_gap_string, gap_fit_cmd, evaluate)

# --------------------------------------------------------------------------------------
# Sobol configuration
# --------------------------------------------------------------------------------------
# Which hyperparameters to VARY in the Sobol design. Everything else uses its "fixed" value
# (from gap_dipole.HYPERPARAMS). To fix a variable (e.g. l_max, alpha_max), remove it here.
# rcut_transition is fixed at 0.5 (the QUIP/soap_turbo literature default) by leaving it out.
SWEEP = ["rcut_hard", "atom_sigma_r", "atom_sigma_t", "n_sparse"]

# Per-species testing: when SPECIES_SPECIFIC is True, each swept parameter listed in INDIVIDUAL is
# varied SEPARATELY for every species (e.g. atom_sigma_r becomes atom_sigma_r__H and atom_sigma_r__O,
# each its own Sobol variable but sharing the base parameter's bounds). Params not in INDIVIDUAL stay
# shared across species. INDIVIDUAL must be a subset of both SWEEP and PER_SPECIES_CAPABLE, so
# strictly-shared knobs like l_max / alpha_max can never be split.
SPECIES_SPECIFIC = False
INDIVIDUAL       = ["atom_sigma_r", "atom_sigma_t"]

N_SOBOL           = 16     # base sample size (power of 2). runs = N*(D+2) if second_order False
CALC_SECOND_ORDER = False  # True -> also pairwise interactions, at N*(2D+2) fits

# Concurrency: run several fits at once, each capped to gap_dipole.THREADS cores. Default fills the
# machine (cores // THREADS): 16 cores / 4 threads -> 4 fits; on Mahti 128/4 -> 32. Overridable with
# --jobs. Watch memory: each concurrent fit holds its own covariance matrix (grows with n_sparse).
MAX_PARALLEL_FITS = max(1, (os.cpu_count() or 1) // core.THREADS)

# Pilot speed knobs (subsample the data so the sweep is cheap; None = use the full sets)
SUBSAMPLE_TRAIN = None
SUBSAMPLE_TEST  = None


def _build_problem():
    """Expand SWEEP into SALib variables, splitting INDIVIDUAL params per species when enabled."""
    individual = INDIVIDUAL if SPECIES_SPECIFIC else []
    bad = [n for n in individual if n not in PER_SPECIES_CAPABLE]
    if bad:
        raise SystemExit(f"INDIVIDUAL entries not per-species-capable: {bad}. "
                         f"Allowed: {PER_SPECIES_CAPABLE}")
    bad = [n for n in individual if n not in SWEEP]
    if bad:
        raise SystemExit(f"INDIVIDUAL entries must also be in SWEEP: {bad}")

    names, bounds, types, scales, fixed = [], [], [], [], {}
    for base in SWEEP:
        spec = HYPERPARAMS[base]
        scale = spec.get("scale", "linear")
        lo, hi = spec["bounds"]
        b = [math.log10(lo), math.log10(hi)] if scale == "log" else [lo, hi]   # sample log-uniformly
        expanded = [species_key(base, Z) for Z in SPECIES_Z] if base in individual else [base]
        for name in expanded:
            names.append(name)
            bounds.append(b)
            types.append(spec["type"])
            scales.append(scale)
            fixed[name] = spec["fixed"]
    return names, bounds, types, scales, fixed


_NAMES, _BOUNDS, _TYPES, _SCALES, _FIXED = _build_problem()

# SALib problem (variable names may include per-species entries like atom_sigma_r__O). `scales` is
# ours (not SALib's): "log" vars are sampled in log10 space and mapped back to the real value.
PROBLEM = {"num_vars": len(_NAMES), "names": _NAMES, "bounds": _BOUNDS, "types": _TYPES,
           "scales": _SCALES}

# Smoke-test nominal = every (expanded) variable at its base parameter's fixed value.
NOMINAL = dict(_FIXED)

RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
SENS_JSON   = os.path.join(BASE_DIR, "sensitivity_indices.json")
BEST_JSON   = os.path.join(BASE_DIR, "best_config.json")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def prepare_dataset():
    """Subsample train/test once for the sweep; return (train_path, test_path)."""
    train, test = TRAIN_FULL, TEST_FULL
    if SUBSAMPLE_TRAIN:
        train = core.subsample(TRAIN_FULL, os.path.join(BASE_DIR, "pilot_train.xyz"), SUBSAMPLE_TRAIN)
        print(f"[data] subsampled train -> {SUBSAMPLE_TRAIN} frames: {train}")
    if SUBSAMPLE_TEST:
        test = core.subsample(TEST_FULL, os.path.join(BASE_DIR, "pilot_test.xyz"), SUBSAMPLE_TEST)
        print(f"[data] subsampled test -> {SUBSAMPLE_TEST} frames: {test}")
    return train, test


def params_from_row(row):
    """Map a SALib sample row to a named param dict: log vars -> 10**val, int vars rounded."""
    p = {}
    for name, val, typ, scale in zip(PROBLEM["names"], row, PROBLEM["types"], PROBLEM["scales"]):
        if scale == "log":
            p[name] = float(10.0 ** val)              # real value; results.csv shows the real value
        else:
            p[name] = int(round(val)) if typ == "int" else float(val)
    return p


def smoke_test(train, test):
    print(f"[smoke] descriptor={DESCRIPTOR}  nominal={NOMINAL}")
    print(f"[smoke] gap string:\n  {build_gap_string(NOMINAL)}\n")
    t0 = time.time()
    rmse, r2, status = evaluate("smoke", NOMINAL, train, test)
    dt = time.time() - t0
    print(f"[smoke] status={status}  RMSE={rmse}  R2={r2}  ({dt:.0f}s)")
    if status != "ok" or not np.isfinite(rmse):
        print("[smoke] FAILED. Inspect runs/smoke/*.log. "
              "If soap_turbo dipole fitting is unsupported, set DESCRIPTOR='soap' in gap_dipole.py.")
        return False
    trimmed = os.path.join(RUNS_DIR, "smoke", "trimmed.xyz")
    preds = np.array([f.calc.results[PRED_KEY] for f in read(trimmed, index=":")])
    if not np.any(np.abs(preds) > 1e-8):
        print("[smoke] FAILED: all predicted dipoles are zero. Model did not learn dipoles.")
        return False
    print(f"[smoke] OK: predicted dipoles are non-zero (|pred| max={np.abs(preds).max():.4f}).")
    return True


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run only the single smoke-test fit")
    ap.add_argument("--n", type=int, default=N_SOBOL,
                    help=f"Sobol base sample size (default {N_SOBOL}); runs = n*(D+2) if 2nd-order off")
    ap.add_argument("--jobs", type=int, default=MAX_PARALLEL_FITS,
                    help=f"concurrent fits, each using gap_dipole.THREADS cores (default {MAX_PARALLEL_FITS})")
    args = ap.parse_args()
    n_base = args.n

    os.makedirs(RUNS_DIR, exist_ok=True)
    train, test = prepare_dataset()

    if not smoke_test(train, test):
        sys.exit(1)
    if args.smoke:
        return

    # Sobol sample
    param_values = sobol_sample.sample(PROBLEM, n_base, calc_second_order=CALC_SECOND_ORDER)
    n_runs = len(param_values)
    jobs = max(1, args.jobs)
    print(f"\n[sobol] {n_runs} evaluations (N={n_base}, D={PROBLEM['num_vars']}, "
          f"second_order={CALC_SECOND_ORDER}) | {jobs} concurrent fits x {core.THREADS} threads\n")

    # Evaluate samples concurrently (each fit isolated in runs/<i>/, capped to THREADS cores).
    # evaluate() runs in worker threads; result handling (CSV/Y) stays on the main thread -> no locks.
    samples = [(i, params_from_row(row)) for i, row in enumerate(param_values)]
    Y = np.full(n_runs, np.nan)
    fieldnames = ["index", "rmse", "r2", "status"] + PROBLEM["names"]
    with open(RESULTS_CSV, "w", newline="") as csvf, \
            concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        w = csv.DictWriter(csvf, fieldnames=fieldnames)
        w.writeheader()
        csvf.flush()
        futs = {ex.submit(evaluate, i, p, train, test): (i, p) for i, p in samples}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            i, p = futs[fut]
            rmse, r2, status = fut.result()
            Y[i] = rmse
            w.writerow({"index": i, "rmse": rmse, "r2": r2, "status": status, **p})
            csvf.flush()
            print(f"[{done}/{n_runs}] idx={i:<3} {status:>16}  RMSE={rmse}")

    # Sensitivity analysis (impute NaN with mean so analyze() runs; flag if too many failed)
    n_fail = int(np.sum(~np.isfinite(Y)))
    if n_fail:
        print(f"\n[warn] {n_fail}/{n_runs} evaluations failed (NaN RMSE); "
              "imputing with mean for sensitivity analysis.")
    if np.all(~np.isfinite(Y)):
        print("[error] every evaluation failed; cannot analyse. Inspect runs/*/*.log.")
        sys.exit(1)
    Y_clean = Y.copy()
    Y_clean[~np.isfinite(Y_clean)] = np.nanmean(Y)

    Si = sobol_analyze.analyze(PROBLEM, Y_clean, calc_second_order=CALC_SECOND_ORDER,
                               print_to_console=False)
    sens = {"names": PROBLEM["names"],
            "S1": list(map(float, Si["S1"])), "S1_conf": list(map(float, Si["S1_conf"])),
            "ST": list(map(float, Si["ST"])), "ST_conf": list(map(float, Si["ST_conf"])),
            "n_runs": n_runs, "n_failed": n_fail, "descriptor": DESCRIPTOR}
    with open(SENS_JSON, "w") as f:
        json.dump(sens, f, indent=2)

    print("\n[sensitivity]  hyperparameter        S1       ST")
    for name, s1, st in sorted(zip(PROBLEM["names"], Si["S1"], Si["ST"]), key=lambda x: -x[2]):
        print(f"               {name:<18} {s1:8.4f} {st:8.4f}")

    # Best config
    best_i = int(np.nanargmin(Y))
    best_p = params_from_row(param_values[best_i])
    best = {"index": best_i, "rmse": float(Y[best_i]), "params": best_p,
            "descriptor": DESCRIPTOR,
            "gap": build_gap_string(best_p),
            "gap_fit_command": " ".join(gap_fit_cmd(train, "gap.xml", build_gap_string(best_p), best_p))}
    with open(BEST_JSON, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n[best] run {best_i}  RMSE={Y[best_i]:.6f}  params={best_p}")
    print(f"[best] written to {BEST_JSON}")
    print(f"[done] results={RESULTS_CSV}  sensitivity={SENS_JSON}")


if __name__ == "__main__":
    main()
