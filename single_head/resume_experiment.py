"""
Resume a partially-completed single-head experiment.

The main pipeline (run_experiment.py) always starts a fresh experiment_<ts> dir and
re-runs every model. This helper instead points at an EXISTING experiment dir, skips
any (model, dataset) whose `<model>_<dataset>.json` is already present, runs only the
missing combinations, and then (re)builds summary.json + plots + tables from the full
set. Tuned hyper-params are reused from the shared cache, so nothing is re-tuned.

Usage:
    python resume_experiment.py --exp-dir experiment_20260702_154507 --datasets mnist
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import json
import argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
_p = HERE
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
sys.path.insert(0, HERE)

import run_experiment as R
from config import *
from shared.utils import load_tuned_params
from shared.metrics import save_results


def load_existing(exp_dir, dataset, model):
    """Reconstruct the in-memory result dict from an already-saved combo JSON,
    matching exactly what run_combination() returns (so plots/tables are identical)."""
    path = os.path.join(exp_dir, f"{model}_{dataset}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "model":       model.upper().replace("_", "-"),
        "dataset":     R.DATASET_LABEL[dataset],
        "hyperparams": data["hyperparams"],
        "per_config":  data["per_config"],
    }


def main():
    ap = argparse.ArgumentParser(description="Resume a single-head experiment")
    ap.add_argument("--exp-dir", required=True,
                    help="experiment dir name (under results/single_head) or an absolute path")
    ap.add_argument("--datasets", nargs="+", default=["mnist"],
                    choices=["mnist", "hhar", "fashion_sw"])
    cli = ap.parse_args()

    exp_dir = cli.exp_dir if os.path.isabs(cli.exp_dir) else os.path.join(RESULTS_DIR, cli.exp_dir)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"experiment dir not found: {exp_dir}")

    datasets = cli.datasets
    t_start  = datetime.now()
    print(f"\n{'#'*70}\n#  RESUME SINGLE-HEAD (DIL) EXPERIMENT\n{'#'*70}", flush=True)
    print(f"  started   : {t_start:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  exp-dir   : {exp_dir}", flush=True)
    print(f"  datasets  : {datasets}", flush=True)
    print(f"  models    : {len(MODELS)}  ->  {MODELS}", flush=True)

    model_dataset_params = load_tuned_params(TUNED_PARAMS_CACHE, datasets, MODELS)

    combos      = [(ds, m) for ds in datasets for m in MODELS]
    all_results = []
    for ds, m in combos:
        existing = load_existing(exp_dir, ds, m)
        if existing is not None:
            print(f"  [skip] {m:<16} on {ds:<10} (already done)", flush=True)
            all_results.append(existing)
            continue
        print(f"  [run ] {m:<16} on {ds:<10} ...", flush=True)
        all_results.append(R.run_combination(ds, m, model_dataset_params, exp_dir, EPOCHS))

    # Rebuild the aggregate artifacts from the full set (existing + newly run).
    exp_ts = os.path.basename(exp_dir).replace("experiment_", "")
    save_results({
        "experiment":      exp_ts,
        "epochs":          EPOCHS,
        "seed":            SEED,
        "datasets":        datasets,
        "subsets":         {d: SUBSET_BY_DATASET[d] for d in datasets},
        "tune_holdouts":   {d: TUNE_SUBSET_BY_DATASET[d] for d in datasets},
        "tune_split_seed": TUNE_SPLIT_SEED,
        "results":         all_results,
    }, os.path.join(exp_dir, "summary.json"))
    R.save_plots(all_results, exp_dir)
    R.print_metrics_table(all_results)
    R.print_hyperparams_table(all_results)

    print(f"\n{'#'*70}\n#  DONE — resume complete\n{'#'*70}", flush=True)
    print(f"  elapsed   : {datetime.now() - t_start}", flush=True)
    print(f"  results   : {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
