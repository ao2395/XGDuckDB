#!/usr/bin/env python3
"""
Random-search tuner for DuckDB RL cardinality model hyperparameters.

It repeatedly runs `run_tpcds_benchmark.py` in a fresh process with different RL_* env vars,
parses the printed Q-error summary, and keeps the best configuration.

Key design points:
- Each trial is a separate process => fresh RLBoostingModel instance.
- Supports `--skip-load` to reuse an existing TPC-DS database (fast).
- Writes a results CSV you can resume from.

Example:
  # First create/load a persistent DB once (slow, done once):
  python3 run_tpcds_benchmark.py --db tpcds_sf1.db --sf 1 --limit 1 --clean

  # Then tune quickly reusing it:
  python3 tune_hyperparams.py --db tpcds_sf1.db --sf 1 --skip-load --limit 50 --trials 50
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


SUMMARY_RE = re.compile(
    r"^RL\s*:\s*median=(?P<median>[\d.]+)\s+p90=(?P<p90>[\d.]+)\s+p95=(?P<p95>[\d.]+)\s*$",
    re.IGNORECASE,
)


def sample_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 1.000001)
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class TrialResult:
    median: float
    p90: float
    p95: float

    def score_tuple(self) -> Tuple[float, float, float]:
        # Lexicographic: prefer lower median, then lower p90, then lower p95
        return (self.median, self.p90, self.p95)


def parse_summary(stdout: str) -> Optional[TrialResult]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = SUMMARY_RE.match(ln)
        if m:
            return TrialResult(
                median=float(m.group("median")),
                p90=float(m.group("p90")),
                p95=float(m.group("p95")),
            )
    return None


def sample_params(rng: random.Random) -> Dict[str, str]:
    # Reasonable search space, biased toward median improvements:
    max_depth = rng.choice([4, 5, 6, 7])
    eta = clamp(sample_log_uniform(rng, 0.03, 0.2), 0.01, 0.3)
    trees_per_update = rng.randint(5, 25)
    subsample = clamp(rng.uniform(0.6, 1.0), 0.4, 1.0)
    colsample = clamp(rng.uniform(0.6, 1.0), 0.4, 1.0)
    min_child_weight = rng.randint(1, 8)

    objective = rng.choice(["reg:absoluteerror", "reg:squarederror"])

    # Regularization knobs (keep small by default)
    lam = clamp(sample_log_uniform(rng, 0.2, 3.0), 0.0, 10.0)
    alpha = clamp(sample_log_uniform(rng, 1e-4, 0.2), 0.0, 2.0)
    gamma = clamp(sample_log_uniform(rng, 1e-5, 0.1), 0.0, 2.0)

    return {
        "RL_MAX_DEPTH": str(max_depth),
        "RL_ETA": f"{eta:.6f}",
        "RL_TREES_PER_UPDATE": str(trees_per_update),
        "RL_SUBSAMPLE": f"{subsample:.3f}",
        "RL_COLSAMPLE_BYTREE": f"{colsample:.3f}",
        "RL_MIN_CHILD_WEIGHT": str(min_child_weight),
        "RL_OBJECTIVE": objective,
        "RL_LAMBDA": f"{lam:.6f}",
        "RL_ALPHA": f"{alpha:.6f}",
        "RL_GAMMA": f"{gamma:.6f}",
    }


def run_trial(
    trial_idx: int,
    params: Dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[int, Optional[TrialResult], str]:
    trial_csv = output_dir / f"trial_{trial_idx:04d}.csv"
    cmd = [
        sys.executable,
        str(Path(args.benchmark_script).resolve()),
        "--duckdb",
        args.duckdb,
        "--db",
        args.db,
        "--sf",
        str(args.sf),
        "--queries",
        args.queries,
        "--output",
        str(trial_csv),
        "--clean",
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.shuffle:
        cmd += ["--shuffle"]
    if args.skip_load:
        cmd += ["--skip-load"]

    env = os.environ.copy()
    env.update(params)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    stdout = proc.stdout

    # If the benchmark failed (non-zero), still capture output for debugging
    result = parse_summary(stdout)
    return proc.returncode, result, stdout


def load_existing_results(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Random-search tuner for DuckDB RL hyperparameters")
    p.add_argument("--trials", type=int, default=30, help="Number of random trials to run")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (0 means time-based)")
    p.add_argument("--limit", type=int, default=50, help="Limit queries per trial (speed vs fidelity)")
    p.add_argument("--sf", type=float, default=1.0, help="TPC-DS scale factor")
    p.add_argument("--db", default="tpcds_sf1.db", help="DuckDB database file to use (persistent strongly recommended)")
    p.add_argument("--skip-load", action="store_true", help="Pass --skip-load to run_tpcds_benchmark.py")
    p.add_argument("--shuffle", action="store_true", help="Shuffle query order (reduces overfitting to order)")
    p.add_argument("--duckdb", default="./build/release/duckdb", help="Path to DuckDB binary")
    p.add_argument("--queries", default="queries.sql", help="Path to queries.sql")
    p.add_argument("--benchmark-script", default="run_tpcds_benchmark.py", help="Path to benchmark runner script")
    p.add_argument("--outdir", default="tuning_runs", help="Directory to store trial CSVs + results log")
    p.add_argument("--resume", action="store_true", help="Resume: append to existing results CSV")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "tuning_results.csv"

    existing = load_existing_results(results_csv) if args.resume else []
    start_idx = len(existing)

    seed = args.seed if args.seed != 0 else int(time.time())
    rng = random.Random(seed)

    # Prepare results CSV
    fieldnames = [
        "trial",
        "returncode",
        "rl_median",
        "rl_p90",
        "rl_p95",
        "params_json",
    ]
    write_header = not results_csv.exists() or not args.resume
    with results_csv.open("a" if args.resume else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        best: Optional[Tuple[TrialResult, Dict[str, str]]] = None

        # If resuming, reconstruct current best
        for row in existing:
            try:
                tr = TrialResult(float(row["rl_median"]), float(row["rl_p90"]), float(row["rl_p95"]))
                params = json.loads(row["params_json"])
                if best is None or tr.score_tuple() < best[0].score_tuple():
                    best = (tr, params)
            except Exception:
                continue

        for i in range(start_idx, start_idx + args.trials):
            params = sample_params(rng)
            rc, res, stdout = run_trial(i, params, args, outdir)

            if res is None:
                # Save full output for debugging
                (outdir / f"trial_{i:04d}.log").write_text(stdout)
                w.writerow(
                    {
                        "trial": i,
                        "returncode": rc,
                        "rl_median": "",
                        "rl_p90": "",
                        "rl_p95": "",
                        "params_json": json.dumps(params, sort_keys=True),
                    }
                )
                f.flush()
                print(f"[{i}] FAILED to parse summary (rc={rc}). Saved log to {outdir}/trial_{i:04d}.log")
                continue

            w.writerow(
                {
                    "trial": i,
                    "returncode": rc,
                    "rl_median": f"{res.median:.6f}",
                    "rl_p90": f"{res.p90:.6f}",
                    "rl_p95": f"{res.p95:.6f}",
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
            f.flush()

            improved = best is None or res.score_tuple() < best[0].score_tuple()
            if improved:
                best = (res, params)
                (outdir / "best_params.json").write_text(json.dumps(params, indent=2, sort_keys=True))

            tag = "BEST" if improved else "ok"
            print(
                f"[{i}] {tag}  RL median={res.median:.2f} p90={res.p90:.2f} p95={res.p95:.2f}  params={params}"
            )

    if best is not None:
        res, params = best
        print("\nBest found:")
        print(f"  RL median={res.median:.4f}  p90={res.p90:.4f}  p95={res.p95:.4f}")
        print("  Export these env vars:")
        for k, v in params.items():
            print(f"    export {k}={v}")
        print(f"\nSaved: {outdir}/best_params.json and {results_csv}")
    else:
        print("No successful trials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


