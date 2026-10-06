"""Run Stage 7 with per-seed checkpoints, save tables, and execute its report notebook."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import sklearn
from robustness import (DEFAULT_SEEDS, SCENARIO_COLUMNS, business_scenarios, run_seed,
                        summarize_replicates, paired_policy_differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--population-size", type=int, default=20_000)
    parser.add_argument("--skip-notebook", action="store_true", help="Compute tables without rendering notebook")
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Provide at least two distinct nonnegative seeds")
    if args.population_size < 100:
        parser.error("population-size must be at least 100")
    config = {"seeds": args.seeds, "population_size": args.population_size,
              "split": [0.70, 0.15, 0.15], "model_seed": 42,
              "scenario_grid": business_scenarios().to_dict("records"),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "source_sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                                for name in ["robustness.py", "simulation.py", "causal_learners.py",
                                             "doubly_robust.py", "policies.py"]}}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "reports" / "stage7_cache" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for position, seed in enumerate(args.seeds, 1):
        path = cache / f"seed_{seed}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            print(f"[{position}/{len(args.seeds)}] Loaded verified-configuration checkpoint: seed {seed}", flush=True)
        else:
            print(f"[{position}/{len(args.seeds)}] Fitting S/T/DR on seed {seed}...", flush=True)
            result = run_seed(seed, population_size=args.population_size)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
            temp.replace(path)
            print(f"[{position}/{len(args.seeds)}] Completed seed {seed}: {len(result['policies'])} policy/scenario rows", flush=True)
        results.append(result)
    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    effects = pd.DataFrame([row for result in results for row in result["effects"]])
    policies = pd.DataFrame([row for result in results for row in result["policies"]])
    diagnostics = pd.DataFrame([row for result in results for row in result["diagnostics"]])
    effect_summary = summarize_replicates(effects, ["model"],
                                          ["ate_error", "pehe", "cate_mae", "cate_correlation"])
    policy_summary = summarize_replicates(policies, [*SCENARIO_COLUMNS, "policy"],
                                          ["incremental_profit", "profit_per_customer", "regret_vs_oracle",
                                           "oracle_profit_captured", "treatment_rate", "dr_incremental_profit",
                                           "observational_estimation_error"])
    paired = paired_policy_differences(policies)
    paired_summary = summarize_replicates(paired, [*SCENARIO_COLUMNS, "comparison"],
                                          ["profit_difference", "left_wins", "tie"])
    outputs = {"stage7_cate_by_seed.csv": effects, "stage7_policy_by_seed.csv": policies,
               "stage7_diagnostics_by_seed.csv": diagnostics, "stage7_cate_summary.csv": effect_summary,
               "stage7_policy_summary.csv": policy_summary, "stage7_paired_by_seed.csv": paired,
               "stage7_paired_summary.csv": paired_summary, "stage7_scenario_grid.csv": business_scenarios()}
    for name, table in outputs.items():
        table.to_csv(tables / name, index=False)
    manifest = {"stage": 7, "experiment_fingerprint": fingerprint, "config": config,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "rows": {name: len(table) for name, table in outputs.items()},
                "interpretation": "Seed SD/quantiles describe Monte Carlo variation; scenarios are paired, not independent. Observational intervals are approximate."}
    (tables / "stage7_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    if not args.skip_notebook:
        import nbformat
        from nbclient import NotebookClient
        path = ROOT / "notebooks" / "05_robustness_and_sensitivity.ipynb"
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.write(notebook, path)
    print(f"Stage 7 complete: {len(args.seeds)} seeds, 27 scenarios, {len(policies)} policy rows.", flush=True)


if __name__ == "__main__":
    main()
