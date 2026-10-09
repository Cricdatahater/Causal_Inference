"""Run Stage 8 with per-seed checkpoints, save tables, and execute its report notebook."""
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
from robustness import summarize_replicates
from causal_stress import DEFAULT_SEEDS, STRESS_SCENARIOS, run_stress, paired_baseline_deltas


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
              "stress_scenarios": list(STRESS_SCENARIOS), "economics": [80, 10, 0.2],
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "source_sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                                for name in ["robustness.py", "simulation.py", "causal_learners.py",
                                             "doubly_robust.py", "policies.py", "stress_simulation.py", "causal_stress.py"]}}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "reports" / "stage8_cache" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(args.seeds)*len(STRESS_SCENARIOS)
    position = 0
    for seed in args.seeds:
        for scenario in STRESS_SCENARIOS:
            position += 1
            path = cache / f"seed_{seed}_{scenario[0].replace(' ', '_')}.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                print(f"[{position}/{total}] Loaded checkpoint: seed {seed}, {scenario[0]}", flush=True)
            else:
                print(f"[{position}/{total}] Fitting seed {seed}, {scenario[0]}...", flush=True)
                result = run_stress(seed, scenario, population_size=args.population_size)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
                temp.replace(path)
                print(f"[{position}/{total}] Completed seed {seed}, {scenario[0]}", flush=True)
            results.append(result)
    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frames = {key: pd.DataFrame([row for result in results for row in result[key]])
              for key in ["effects", "policies", "diagnostics", "clipping", "overlap"]}
    effects, policies, diagnostics = (frames[key] for key in ["effects", "policies", "diagnostics"])
    effect_summary = summarize_replicates(effects, ["scenario", "model"],
                                          ["ate_error", "pehe", "cate_mae", "cate_correlation"])
    policy_summary = summarize_replicates(policies, ["scenario", "policy"],
                                          ["incremental_profit", "oracle_profit_captured", "regret_vs_oracle",
                                           "estimated_profit_error", "analytic_policy_identification_gap"])
    diagnostic_summary = summarize_replicates(diagnostics, ["scenario"],
         ["true_ate", "estimated_aipw_ate", "reference_aipw_ate", "analytic_identification_gap",
          "true_weak_overlap_fraction", "estimated_weak_overlap_fraction", "clipped_weight_ess",
          "trimmed_true_ate", "trimmed_estimated_aipw_ate"])
    effect_deltas = paired_baseline_deltas(effects, "pehe", ["model"])
    policy_deltas = paired_baseline_deltas(policies, "incremental_profit", ["policy"])
    outputs = {"stage8_effects_by_seed.csv": effects, "stage8_policies_by_seed.csv": policies,
               "stage8_diagnostics_by_seed.csv": diagnostics, "stage8_clipping_by_seed.csv": frames["clipping"],
               "stage8_overlap_histograms.csv": frames["overlap"], "stage8_effect_summary.csv": effect_summary,
               "stage8_policy_summary.csv": policy_summary, "stage8_diagnostic_summary.csv": diagnostic_summary,
               "stage8_effect_deltas.csv": effect_deltas, "stage8_policy_deltas.csv": policy_deltas}
    for name, table in outputs.items():
        table.to_csv(tables / name, index=False)
    manifest = {"stage": 8, "experiment_fingerprint": fingerprint, "config": config,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "rows": {name: len(table) for name, table in outputs.items()},
                "interpretation": "Scenarios are seed-paired. Hidden U is marginalized in causal truth. Reference observational nuisances do not restore identification."}
    (tables / "stage8_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    if not args.skip_notebook:
        import nbformat
        from nbclient import NotebookClient
        path = ROOT / "notebooks" / "06_causal_stress_tests.ipynb"
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.write(notebook, path)
    print(f"Stage 8 complete: {len(args.seeds)} seeds, {len(STRESS_SCENARIOS)} stress scenarios, {len(policies)} policy rows.", flush=True)


if __name__ == "__main__":
    main()
