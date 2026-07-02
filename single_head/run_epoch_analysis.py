"""
Single-Head (DIL) -- Epoch-sensitivity analysis.

Re-runs the same hyperparameter tuning as run_experiment.py, then sweeps EPOCH_NUMBERS on the first CONFIGS task configs to chart each
metric vs. training length.

"""
import os
import argparse
from datetime import datetime
from config import (
    DATASETS, MODELS, DEVICE, RESULTS_DIR, SEED, TUNED_PARAMS_CACHE,
    SUBSET_BY_DATASET, TUNE_SUBSET_BY_DATASET, TUNE_SPLIT_SEED, DATASET_CONFIGS,
)
from run_experiment import run_combination, save_results, save_plots
from shared.utils import load_tuned_params
from shared.epoch_analysis import save_epoch_analysis_plots

EPOCH_NUMBERS = [5, 10, 20]
CONFIGS = 3 # number of configs to run for the epoch analysis experiment

def run_epoch_analysis(model_dataset_params, datasets):
    exp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"epoch_analysis_{exp_ts}")
    os.makedirs(exp_dir, exist_ok=True)

    combos = [(ds, m) for ds in datasets for m in MODELS]
    sweep_dir = os.path.join(exp_dir, "epoch_sweep")
    sweep_cfgs = {ds: list(DATASET_CONFIGS[ds])[:CONFIGS]
                  for ds in datasets if DATASET_CONFIGS.get(ds)}
    epoch_dirs = {E: os.path.join(sweep_dir, f"epochs_{E}") for E in EPOCH_NUMBERS}
    for d in epoch_dirs.values():
        os.makedirs(d, exist_ok=True)

    results_by_epoch = {}
    for E in EPOCH_NUMBERS:
        results = []
        for ds, m in combos:
            results.append(run_combination(ds, m, model_dataset_params, epoch_dirs[E], E, sweep_cfgs.get(ds)))
        results_by_epoch[E] = results

        save_results({
            "experiment":        exp_ts,
            "epochs":            E,
            "seed":              SEED,
            "subsets":           {d: SUBSET_BY_DATASET[d] for d in datasets},
            "tune_holdouts":     {d: TUNE_SUBSET_BY_DATASET[d] for d in datasets},
            "tune_split_seed":   TUNE_SPLIT_SEED,
            "results":           results,
        }, os.path.join(epoch_dirs[E], "summary.json"))
        save_plots(results, epoch_dirs[E])

    # Cross-epoch analysis: each metric vs epochs, one line per model, per dataset.
    save_epoch_analysis_plots(results_by_epoch, os.path.join(exp_dir, "epoch_analysis"))
    return exp_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single-Head (DIL) epoch-number model performance analysis")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=["mnist", "hhar", "fashion_sw"])
    cli = ap.parse_args()
    datasets = cli.datasets

    t_start = datetime.now()
    print(f"\n{'#'*70}\n#  SINGLE-HEAD (DIL) EPOCH ANALYSIS\n{'#'*70}", flush=True)
    print(f"  started   : {t_start:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  device    : {DEVICE}", flush=True)
    print(f"  datasets  : {datasets}", flush=True)
    print(f"  models    : {len(MODELS)}  ->  {MODELS}", flush=True)

    model_dataset_params = load_tuned_params(TUNED_PARAMS_CACHE, datasets, MODELS, SEED)
    # Epoch sweep + cross-epoch analysis plots.
    out_dir = run_epoch_analysis(model_dataset_params, datasets)

    elapsed = datetime.now() - t_start
    print(f"\n{'#'*70}\n#  DONE — epoch analysis complete\n{'#'*70}", flush=True)
    print(f"  elapsed   : {elapsed}", flush=True)
    print(f"  results   : {out_dir}", flush=True)
