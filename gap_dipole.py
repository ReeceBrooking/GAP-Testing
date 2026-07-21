#!/usr/bin/env python
"""
Core GAP dipole-fitting pipeline for a SINGLE hyperparameter configuration.

This module holds everything needed to fit + score one model:
    gap_fit -> quip -> trim.py -> errorcalc.py
plus the configuration (paths, dataset, descriptor, hyperparameter table) shared with the
Sobol driver (sobol_dipole.py imports from here, so there is one source of truth).

Run ONE configuration directly (useful for manual runs and SLURM array tasks):

    # all hyperparameters at their fixed defaults, into runs/single/
    python gap_dipole.py --index single

    # override individual hyperparameters (any not given use their HYPERPARAMS "fixed" value)
    python gap_dipole.py --index 7 --set rcut_hard 3.0 --set l_max 5 --set alpha_max 5

    # or supply a JSON dict of overrides (e.g. one row of a Sobol design)
    python gap_dipole.py --index 42 --params-json my_params.json

    # point at specific data
    python gap_dipole.py --index 3 --train train.xyz --test test.xyz

Prints a one-line result and writes runs/<index>/result.json. Exit code 0 iff status == "ok".
Run under the quipenv interpreter (needs ase, numpy). gap_fit/quip are the local QUIP binaries.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from ase.io import read, write
from ase.data import chemical_symbols

# --------------------------------------------------------------------------------------
# Configuration (edit here)
# --------------------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # this project dir (portable: move it/run anywhere)
# gap_fit/quip binaries are OUTSIDE the project and machine-specific -> override with QUIP_BIN env var
# (e.g. on Mahti: export QUIP_BIN=/path/to/QUIP/build/.../Programs).
QUIP_PROG  = os.environ.get("QUIP_BIN", "/home/reece/QUIP/builddir/src/Programs")
GAP_FIT    = os.path.join(QUIP_PROG, "gap_fit")
QUIP       = os.path.join(QUIP_PROG, "quip")
PYTHON     = sys.executable                       # this quipenv interpreter (for trim.py/errorcalc.py)
TRIM_PY    = os.path.join(BASE_DIR, "trim.py")
ERRORCALC  = os.path.join(BASE_DIR, "errorcalc.py")

# Dataset (waterbulk)
TRAIN_FULL = os.path.join(BASE_DIR, "waterbulk", "train_waterbulk.xyz")
TEST_FULL  = os.path.join(BASE_DIR, "waterbulk", "test_waterbulk.xyz")
REF_KEY    = "mu"        # reference total dipole key in the xyz header (info)
PRED_KEY   = "dipole"    # predicted total dipole key (ASE calc.results)
SPECIES_Z  = [1, 8]      # H, O -> n_species=2; central_index runs 1..n_species

# Descriptor choice: "soap_turbo" (preferred) or "soap" (fallback, known-working)
DESCRIPTOR = "soap_turbo"

# soap_turbo descriptor compression (dimensionality reduction). One of:
#   "trivial"                -> pivot recipe: compressed dim ~ linear in n_species*alpha_max (default)
#   "none" / ""              -> NO compression: full power spectrum (largest, most expressive, slowest)
#   "0_0".."2_2"             -> Darby nu_R_nu_S recipes (REQUIRE equal alpha_max across all species)
COMPRESS_MODE = "trivial"

# Sparse-point selection method (gap_fit's own default is RANDOM; this workflow uses CUR). One of:
#   "cur_points"    -> CUR on the descriptor matrix: leverage-score (SVD) selection (default here)
#   "cur_covariance"-> CUR on the kernel matrix
#   "kmeans"/"cluster" -> coverage-oriented, outlier-robust
#   "random"        -> gap_fit's default; follows data density; cheapest
#   "index_file"    -> read 1-based sparse indices from SPARSE_FILE (use this for external FPS)
# For externally-computed Farthest-Point Sampling: set SPARSE_METHOD="index_file" and SPARSE_FILE.
SPARSE_METHOD = "cur_points"
SPARSE_FILE   = ""          # path to sparse-index file, only used when SPARSE_METHOD="index_file"

# gap_fit random seed for CUR sparse-point selection. None -> gap_fit default (-1 = fresh each run,
# NON-reproducible: two identical configs give slightly different fits/R2). Set an int for
# DETERMINISTIC, reproducible fits. Overridable per single run with --seed.
RND_SEED = None

# Threads per gap_fit/quip process. gap_fit has no OpenMP here, so parallelism comes only from
# threaded OpenBLAS (the SVD/solve); this caps it so concurrent fits don't oversubscribe. Set it to
# the cores you want EACH fit to use; the Sobol sweep then runs (cores // THREADS) fits at once.
THREADS = 4

# Per-run timeouts
FIT_TIMEOUT  = 3600      # seconds per gap_fit
PRED_TIMEOUT = 1800      # seconds per quip prediction

RUNS_DIR = os.path.join(BASE_DIR, "runs")

# --------------------------------------------------------------------------------------
# Hyperparameters: single source of truth.
#   fixed  = value used when the parameter is NOT provided (the default)
#   bounds = [lo, hi] Sobol sampling range when the parameter IS swept (used by sobol_dipole.py)
#   type   = "int" (rounded to nearest integer when building the command) or "float"
# `rcut_transition` is a helper: rcut_soft = rcut_hard - rcut_transition (keeps soft < hard);
# it equals soap_turbo's cutoff_transition_width (QUIP/literature default 0.5).
# --------------------------------------------------------------------------------------
HYPERPARAMS = {
    "rcut_hard":            {"fixed": 4.0, "bounds": [2.0, 8.0],   "type": "float"},
    "rcut_transition":      {"fixed": 0.5, "bounds": [0.3, 1.0],   "type": "float"},
    "l_max":                {"fixed": 7,   "bounds": [2, 7],       "type": "int"},
    "alpha_max":            {"fixed": 7,   "bounds": [4, 7],      "type": "int"},
    "atom_sigma_r":         {"fixed": 0.5, "bounds": [0.1, 0.6],   "type": "float"},
    "atom_sigma_t":         {"fixed": 0.5, "bounds": [0.1, 0.6],   "type": "float"},
    "atom_sigma_r_scaling": {"fixed": 0.0, "bounds": [0.0, 0.5],   "type": "float"},
    "atom_sigma_t_scaling": {"fixed": 0.0, "bounds": [0.0, 0.5],   "type": "float"},
    "amplitude_scaling":    {"fixed": 1.0, "bounds": [0.0, 4.0],   "type": "float"},
    "central_weight":       {"fixed": 1.0, "bounds": [0.5, 2.0],   "type": "float"},
    "radial_enhancement":   {"fixed": 0,   "bounds": [0, 2],       "type": "int"},
    "zeta":                 {"fixed": 2,   "bounds": [1, 4],       "type": "int"},
    "delta":                {"fixed": 1.0, "bounds": [0.1, 2.0],   "type": "float"},
    "n_sparse":             {"fixed": 300, "bounds": [100, 5000],  "type": "int"},
    # gap_fit-level dipole regularisation (default_dipole_sigma; gap_fit's own default is 0.001).
    # "scale": "log" -> the Sobol sampler draws it LOG-uniformly across the bounds (regularisation
    # spans orders of magnitude); results.csv still stores the real value. Applied in gap_fit_cmd,
    # NOT in the descriptor string. Not per-species.
    "dipole_sigma":         {"fixed": 0.00001, "bounds": [1e-5, 1e-3], "type": "float", "scale": "log"},
}

# soap_turbo hyperparameters that are genuinely per-NEIGHBOUR-species arrays and so MAY be given a
# different value per species (see SPECIES_SPECIFIC / INDIVIDUAL in sobol_dipole.py, or --set
# name__<Symbol> on the CLI). Everything else (l_max, alpha_max, rcut_*, radial_enhancement, and the
# GAP-level zeta/delta/n_sparse) is scalar and ALWAYS shared across species.
PER_SPECIES_CAPABLE = ["atom_sigma_r", "atom_sigma_t", "atom_sigma_r_scaling",
                       "atom_sigma_t_scaling", "amplitude_scaling", "central_weight"]


# --------------------------------------------------------------------------------------
# Parameter resolution (shared scalars + optional per-species arrays)
# --------------------------------------------------------------------------------------
def sp_symbol(Z):
    """Element symbol for atomic number Z (e.g. 8 -> 'O'), used to name per-species variables."""
    return chemical_symbols[Z]


def species_key(base, Z):
    """Per-species variable name, e.g. species_key('atom_sigma_r', 8) -> 'atom_sigma_r__O'."""
    return f"{base}__{sp_symbol(Z)}"


def valid_param(name):
    """True if `name` is a known shared hyperparameter or a valid per-species variant base__Symbol."""
    if name in HYPERPARAMS:
        return True
    if "__" in name:
        base, sym = name.rsplit("__", 1)
        return base in PER_SPECIES_CAPABLE and sym in [sp_symbol(Z) for Z in SPECIES_Z]
    return False


def _cast(spec, val):
    return int(round(val)) if spec["type"] == "int" else float(val)


def param_value(name, p):
    """Scalar value of a shared parameter: p[name] over its fixed default, int-cast if needed."""
    spec = HYPERPARAMS[name]
    return _cast(spec, p[name] if name in p else spec["fixed"])


def species_array(base, p):
    """List of len(SPECIES_Z) values for a per-species-capable param, in SPECIES_Z order.

    Precedence per species Z: the per-species key `base__<Symbol>` if present, else the shared
    key `base`, else the fixed default. Floats are rounded to 4 dp for a clean gap string.
    """
    spec = HYPERPARAMS[base]
    out = []
    for Z in SPECIES_Z:
        key = species_key(base, Z)
        val = p[key] if key in p else (p[base] if base in p else spec["fixed"])
        out.append(_cast(spec, val) if spec["type"] == "int" else round(float(val), 4))
    return out


# --------------------------------------------------------------------------------------
# Descriptor / command construction
# --------------------------------------------------------------------------------------
def brace(values):
    """Format a per-species array for a soap_turbo descriptor as a DOUBLE-braced group.

    The arrays live inside the outer `gap={...}`, which consumes one level of braces, so an
    inner array must be written `{{a b}}` to survive as `{a b}` when the descriptor is re-parsed
    by soap_turbo_initialise. (Single braces arrive as `a b` -> "wrong number of value fields".)
    """
    return "{{" + " ".join(str(v) for v in values) + "}}"


def resolve(p):
    """Human-readable resolved view: shared params -> scalar, per-species params -> [per-Z, ...]."""
    q = {}
    for name in HYPERPARAMS:
        q[name] = species_array(name, p) if name in PER_SPECIES_CAPABLE else param_value(name, p)
    return q


def build_gap_string(p):
    """Build the gap={...} descriptor string for one hyperparameter dict `p` (partial ok).

    Shared scalars come from param_value(); per-species arrays (atom_sigma_r etc.) come from
    species_array(), which uses a `base__<Symbol>` override if present else the shared/fixed value.
    """
    L   = param_value("l_max", p)
    A   = param_value("alpha_max", p)
    Rh  = param_value("rcut_hard", p)
    Rs  = round(Rh - param_value("rcut_transition", p), 4)   # rcut_soft = rcut_hard - transition
    ns  = len(SPECIES_Z)

    # sparse-point selection (+ optional index file for external FPS)
    sparse = f"sparse_method={SPARSE_METHOD}"
    if SPARSE_METHOD.lower() == "index_file" and SPARSE_FILE:
        sparse += f" sparse_file={SPARSE_FILE}"

    if DESCRIPTOR == "soap":
        # Single soap descriptor (fallback). atom_sigma is a scalar (species arrays not supported).
        return ("{soap "
                f"cutoff={Rh} n_max={A} l_max={L} atom_sigma={species_array('atom_sigma_r', p)[0]} "
                f"delta={param_value('delta', p)} zeta={param_value('zeta', p)} "
                f"n_sparse={param_value('n_sparse', p)} covariance_type=dot_product {sparse}}}")

    # optional compression (omit the key entirely -> full, uncompressed descriptor)
    compress = "" if COMPRESS_MODE.lower() in ("", "none") else f"compress_mode={COMPRESS_MODE} "

    # soap_turbo: one descriptor per central species, colon-separated inside gap={ ... }
    descs = []
    for ci in range(1, ns + 1):
        d = (
            f"soap_turbo l_max={L} "
            f"alpha_max={brace([A] * ns)} "                                  # shared (broadcast)
            f"atom_sigma_r={brace(species_array('atom_sigma_r', p))} "        # per-species-capable
            f"atom_sigma_t={brace(species_array('atom_sigma_t', p))} "
            f"atom_sigma_r_scaling={brace(species_array('atom_sigma_r_scaling', p))} "
            f"atom_sigma_t_scaling={brace(species_array('atom_sigma_t_scaling', p))} "
            f"amplitude_scaling={brace(species_array('amplitude_scaling', p))} "
            f"central_weight={brace(species_array('central_weight', p))} "
            f"rcut_hard={Rh} rcut_soft={Rs} "
            f"radial_enhancement={param_value('radial_enhancement', p)} "
            "basis=poly3gauss scaling_mode=polynomial add_species=F "
            f"{compress}"
            f"n_species={ns} species_Z={brace(SPECIES_Z)} central_index={ci} "
            f"zeta={param_value('zeta', p)} delta={param_value('delta', p)} "
            f"n_sparse={param_value('n_sparse', p)} "
            f"f0=0.0 covariance_type=dot_product {sparse}"
        )
        descs.append(d)
    return "{" + " : ".join(descs) + "}"


def gap_fit_cmd(train, gp_file, gap_str, p=None):
    """gap_fit command. `p` supplies gap-fit-level hyperparameters (currently dipole_sigma)."""
    dipole_sigma = param_value("dipole_sigma", p or {})   # regularisation of the dipole target
    cmd = [
        GAP_FIT,
        f"at_file={train}",
        f"dipole_parameter_name={REF_KEY}",
        f"gap={gap_str}",
        "default_sigma={1 1 1 1}",
        f"default_dipole_sigma={dipole_sigma:.10g}",
        "e0=0",
        f"gp_file={gp_file}",
    ]
    if RND_SEED is not None:                               # deterministic CUR selection -> reproducible
        cmd.append(f"rnd_seed={int(RND_SEED)}")
    return cmd


def quip_cmd(test, param):
    return [
        QUIP,
        f"atoms_filename={test}",
        f"param_filename={param}",
        "e",
        f"calc_args=dipole={PRED_KEY} local_dipole=local_dipole",
        # Send quip's diagnostic log (Energy=, Cell, banners) to a file so STDOUT carries ONLY the
        # 'AT'-prefixed atom lines. quip writes atoms to stdout via a separate I/O channel from
        # mainlog; if both hit stdout they race and splice a diagnostic mid-line, corrupting a
        # header and breaking trim.py/ASE. Relative path -> lands in the run dir (quip cwd=rundir).
        "output_file=quip_diag.log",
    ]


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------
def capped_env():
    """Process env with BLAS/OpenMP threads capped to THREADS (prevents oversubscription)."""
    t = str(THREADS)
    return dict(os.environ, OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t,
                MKL_NUM_THREADS=t, NUMEXPR_NUM_THREADS=t)


def run(cmd, rundir, log_name, timeout, stdout_path=None):
    """Run a subprocess in rundir; tee stderr(+stdout) to a log file. Return CompletedProcess."""
    log_path = os.path.join(rundir, log_name)
    with open(log_path, "w") as logf:
        if stdout_path:
            with open(stdout_path, "w") as out:
                return subprocess.run(cmd, cwd=rundir, stdout=out, stderr=logf,
                                      timeout=timeout, check=False, env=capped_env())
        return subprocess.run(cmd, cwd=rundir, stdout=logf, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False, env=capped_env())


def parse_scores(errorcalc_stdout):
    """Extract RMSE and R2 floats from errorcalc.py output."""
    rmse = r2 = float("nan")
    for line in errorcalc_stdout.splitlines():
        s = line.strip()
        if s.startswith("RMSE:"):
            rmse = float(s.split(":", 1)[1].split()[0])
        elif s.startswith("R2:"):
            r2 = float(s.split(":", 1)[1].split()[0])
    return rmse, r2


def evaluate(idx, params, train, test):
    """Run the full fit->predict->trim->score pipeline for one config. Returns (rmse, r2, status).

    `idx` names the working dir runs/<idx>/. `params` is a (possibly partial) hyperparameter dict;
    anything missing falls back to its HYPERPARAMS "fixed" value via build_gap_string/resolve.
    """
    rundir = os.path.join(RUNS_DIR, str(idx))
    os.makedirs(rundir, exist_ok=True)
    gp_file   = os.path.join(rundir, "gap.xml")
    out_xyz   = os.path.join(rundir, "out.xyz")
    trim_xyz  = os.path.join(rundir, "trimmed.xyz")
    gap_str   = build_gap_string(params)

    # record the exact command for reproducibility
    with open(os.path.join(rundir, "config.json"), "w") as f:
        json.dump({"index": idx, "params": params, "descriptor": DESCRIPTOR,
                   "gap": gap_str}, f, indent=2)

    try:
        r = run(gap_fit_cmd(train, gp_file, gap_str, params), rundir, "gap_fit.log", FIT_TIMEOUT)
        if r.returncode != 0 or not os.path.exists(gp_file):
            return float("nan"), float("nan"), "fit_failed"

        r = run(quip_cmd(test, gp_file), rundir, "quip.log", PRED_TIMEOUT, stdout_path=out_xyz)
        if r.returncode != 0 or not os.path.exists(out_xyz):
            return float("nan"), float("nan"), "predict_failed"

        subprocess.run([PYTHON, TRIM_PY, out_xyz, trim_xyz], check=True,
                       stdout=subprocess.DEVNULL, timeout=300, env=capped_env())

        sc = subprocess.run([PYTHON, ERRORCALC, trim_xyz, REF_KEY, PRED_KEY],
                            capture_output=True, text=True, timeout=600, env=capped_env())
        with open(os.path.join(rundir, "errorcalc.log"), "w") as f:
            f.write(sc.stdout + "\n" + sc.stderr)
        if sc.returncode != 0:
            return float("nan"), float("nan"), "score_failed"

        rmse, r2 = parse_scores(sc.stdout)
        status = "ok" if np.isfinite(rmse) else "score_parse_failed"
        return rmse, r2, status
    except subprocess.TimeoutExpired:
        return float("nan"), float("nan"), "timeout"
    except Exception as e:  # noqa: BLE001 - log and continue
        with open(os.path.join(rundir, "error.log"), "w") as f:
            f.write(repr(e))
        return float("nan"), float("nan"), "exception"


def subsample(src, dst, n):
    """Write the first n frames of src to dst; return dst."""
    write(dst, read(src, index=f":{n}"))
    return dst


# --------------------------------------------------------------------------------------
# Single-run CLI
# --------------------------------------------------------------------------------------
def parse_overrides(set_pairs, params_json):
    """Build a hyperparameter override dict from --set pairs and/or a --params-json file."""
    overrides = {}
    if params_json:
        with open(params_json) as f:
            overrides.update(json.load(f))
    for name, value in set_pairs:
        overrides[name] = value
    # validate names and cast to float (ints are rounded later). Per-species keys look like
    # `atom_sigma_r__O` (base must be in PER_SPECIES_CAPABLE, symbol in SPECIES_Z).
    clean = {}
    for name, value in overrides.items():
        if not valid_param(name):
            raise SystemExit(
                f"unknown hyperparameter '{name}'. Known shared: {list(HYPERPARAMS)}; "
                f"per-species: {[species_key(b, Z) for b in PER_SPECIES_CAPABLE for Z in SPECIES_Z]}")
        clean[name] = float(value)
    return clean


def main():
    global RND_SEED               # declared before any read of RND_SEED below
    ap = argparse.ArgumentParser(description="Fit + score one GAP dipole model.")
    ap.add_argument("--index", default="single", help="run id -> runs/<index>/ (default: single)")
    ap.add_argument("--set", nargs=2, metavar=("NAME", "VALUE"), action="append", default=[],
                    help="override one hyperparameter (repeatable), e.g. --set rcut_hard 3.5; "
                         "per-species e.g. --set atom_sigma_r__O 0.4 --set atom_sigma_r__H 0.2")
    ap.add_argument("--params-json", help="JSON file with a {name: value} override dict")
    ap.add_argument("--train", default=TRAIN_FULL, help=f"training xyz (default {TRAIN_FULL})")
    ap.add_argument("--test", default=TEST_FULL, help=f"test xyz (default {TEST_FULL})")
    ap.add_argument("--seed", type=int, default=RND_SEED,
                    help="gap_fit rnd_seed for reproducible CUR (default: config RND_SEED)")
    args = ap.parse_args()
    RND_SEED = args.seed          # gap_fit_cmd reads this module global

    overrides = parse_overrides(args.set, args.params_json)
    rundir = os.path.join(RUNS_DIR, str(args.index))
    os.makedirs(rundir, exist_ok=True)

    train, test = args.train, args.test

    resolved = resolve(overrides)
    print(f"[run {args.index}] descriptor={DESCRIPTOR} compress={COMPRESS_MODE} "
          f"sparse={SPARSE_METHOD}")
    print(f"[run {args.index}] overrides={overrides}")
    t0 = time.time()
    rmse, r2, status = evaluate(args.index, overrides, train, test)
    dt = time.time() - t0

    result = {"index": args.index, "status": status, "rmse": rmse, "r2": r2,
              "seconds": round(dt, 1), "overrides": overrides, "resolved": resolved,
              "descriptor": DESCRIPTOR, "compress_mode": COMPRESS_MODE, "sparse_method": SPARSE_METHOD,
              "train": train, "test": test, "gap": build_gap_string(overrides)}
    with open(os.path.join(rundir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[run {args.index}] status={status}  RMSE={rmse}  R2={r2}  ({dt:.0f}s)")
    print(f"[run {args.index}] -> {os.path.join(rundir, 'result.json')}")
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
