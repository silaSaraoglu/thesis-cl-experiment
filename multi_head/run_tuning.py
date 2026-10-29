"""
Multi-Head (TIL) -- Phase-1 tuning, run once and cached.

This script owns ALL tuning logic. It tunes the selected datasets and writes the best
hyperparameters to tuned_params.json. run_experiment.py and run_epoch_analysis.py only READ
that cache -- they contain no tuning code -- so tuning happens here once.

Usage:
    python run_tuning.py                      # tune all datasets, cache them
    python run_tuning.py --datasets hhar      # tune just one dataset
"""
import warnings
warnings.filterwarnings("ignore")  # silence import-time lib warnings (jaxopt/Avalanche/torch)
                                   # before optuna/run_experiment are imported below
import argparse
from argparse import Namespace
from datetime import datetime
import numpy as np
import optuna
import run_experiment as run_experiment   # only for the model run
from config import (
    DATASETS, MODELS, DEVICE, SEED, TUNED_PARAMS_CACHE,
    MNIST_DIR, WISDM_DIR, HHAR_DIR, FASHION_DIR,
    TUNE_SUBSET_MNIST, TUNE_SUBSET_WISDM, TUNE_SUBSET_HHAR, TUNE_SUBSET_FASHION,
    TUNE_SPLIT_SEED,
)
from shared.utils import load_cache, save_cache
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 15
TUNE_EPOCHS = 5

# GIM on MNIST: original paper hyperparameters-- remains as fixed
GIM_MNIST_PARAMS = dict(
    batch_size              = 32,
    learning_rate           = 1e-4,
    hidden_size_rnn         = 128,
    hidden_size_autoencoder = 500,
)


# Optuna param samplers
# One search space per model family, reused for every dataset (MNIST / WISDM / HHAR / Fashion-SW).
# Each (model, dataset) study is tuned independently, so the saved best values differ.
def gim_params(trial):
    return dict(
        hidden_size_rnn         = trial.suggest_categorical("hidden_size_rnn", [32, 64, 128, 256]),
        hidden_size_autoencoder = trial.suggest_categorical("hidden_size_autoencoder", [300, 400, 500]),
        learning_rate           = trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        batch_size              = trial.suggest_categorical("batch_size", [16, 32, 64]),
    )

# Joint-LSTM shares the LSTM search space.
def lstm_params(trial):
    return dict(
        hidden_size_rnn = trial.suggest_categorical("hidden_size_rnn", [32, 64, 128, 256]),
        learning_rate   = trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        batch_size      = trial.suggest_categorical("batch_size", [16, 32, 64]),
    )

def esn_base_params(trial):
    return dict(
        esn_units     = trial.suggest_categorical("esn_units", [300, 500, 1000]),
        learning_rate = trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64]),
    )

# Arg-builder + objective factory for tuning
def to_args(extra, **kwargs):
    return Namespace(device=DEVICE, **extra, **kwargs)

tuned = {}
def make_objective(runner, arg_fn, param_fn, data_dir, pass_trial=True):
    """Return an Optuna objective that samples params, runs the experiment, and
    returns mean val accuracy."""
    tune_n = (TUNE_SUBSET_MNIST  if data_dir == MNIST_DIR
              else TUNE_SUBSET_WISDM if data_dir == WISDM_DIR
              else TUNE_SUBSET_HHAR if data_dir == HHAR_DIR
              else TUNE_SUBSET_FASHION)
    def objective(trial):
        params = param_fn(trial)
        args = arg_fn({**params, "epochs": TUNE_EPOCHS}, data_dir=data_dir, subset=None,
                      holdout_n=tune_n, holdout_seed=TUNE_SPLIT_SEED, use_holdout=True)
        trial_kwargs = {"trial": trial} if pass_trial else {}
        return float(np.mean(runner(args, False, **trial_kwargs)[0]))
    return objective


def tune(label, objective):
    n = N_TRIALS
    print(f"  Tuning  {label:<40} ({n} trials) ...", flush=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=min(5, n),
            n_warmup_steps=1,
        ),
    )
    study.optimize(objective, n_trials=n, show_progress_bar=False)
    print(f"  Done    {label:<40} best={study.best_value:.4f}", flush=True)
    return study.best_params


TUNE_STUDIES = [
    ("GIM-ALSTM on WISDM",   make_objective(lambda a,v,**kw: run_experiment.run_gim_wisdm_mh("alstm",a,v,**kw), to_args, gim_params,WISDM_DIR)),
    ("LSTM on MNIST",        make_objective(run_experiment.run_naive_lstm_mnist_mh, to_args, lstm_params, MNIST_DIR)),
    ("LSTM on WISDM",        make_objective(run_experiment.run_naive_lstm_wisdm_mh, to_args, lstm_params, WISDM_DIR)),
    ("Joint-LSTM on MNIST",  make_objective(run_experiment.run_joint_lstm_mnist_mh, to_args, lstm_params, MNIST_DIR, pass_trial=False)),
    ("Joint-LSTM on WISDM",  make_objective(run_experiment.run_joint_lstm_wisdm_mh, to_args, lstm_params, WISDM_DIR, pass_trial=False)),
    ("LSTM-Replay on MNIST", make_objective(run_experiment.run_lstm_replay_mnist_mh, to_args, lstm_params, MNIST_DIR)),
    ("LSTM-Replay on WISDM", make_objective(run_experiment.run_lstm_replay_wisdm_mh, to_args, lstm_params, WISDM_DIR)),
    ("ESN-Base on MNIST",    make_objective(run_experiment.run_esn_naive_mnist_mh, to_args, esn_base_params, MNIST_DIR)),
    ("ESN-Base on WISDM",    make_objective(run_experiment.run_esn_naive_wisdm_mh, to_args, esn_base_params, WISDM_DIR)),
    ("Joint-ESN on MNIST",   make_objective(run_experiment.run_joint_esn_mnist_mh, to_args, esn_base_params, MNIST_DIR, pass_trial=False)),
    ("Joint-ESN on WISDM",   make_objective(run_experiment.run_joint_esn_wisdm_mh, to_args, esn_base_params, WISDM_DIR, pass_trial=False)),
    ("GIM-ALSTM on HHAR",    make_objective(lambda a,v,**kw: run_experiment.run_gim_hhar_mh("alstm",a,v,**kw), to_args, gim_params, HHAR_DIR)),
    ("LSTM on HHAR",         make_objective(run_experiment.run_naive_lstm_hhar_mh, to_args, lstm_params, HHAR_DIR)),
    ("Joint-LSTM on HHAR",   make_objective(run_experiment.run_joint_lstm_hhar_mh, to_args, lstm_params, HHAR_DIR, pass_trial=False)),
    ("LSTM-Replay on HHAR",  make_objective(run_experiment.run_lstm_replay_hhar_mh, to_args, lstm_params, HHAR_DIR)),
    ("ESN-Base on HHAR",     make_objective(run_experiment.run_esn_naive_hhar_mh, to_args, esn_base_params, HHAR_DIR)),
    ("Joint-ESN on HHAR",    make_objective(run_experiment.run_joint_esn_hhar_mh, to_args, esn_base_params, HHAR_DIR, pass_trial=False)),
    ("GIM-ALSTM on Fashion-SW", make_objective(lambda a,v,**kw: run_experiment.run_gim_fashionsw_mh("alstm",a,v,**kw), to_args, gim_params, FASHION_DIR)),
    ("LSTM on Fashion-SW",       make_objective(run_experiment.run_naive_lstm_fashionsw_mh, to_args, lstm_params, FASHION_DIR)),
    ("Joint-LSTM on Fashion-SW", make_objective(run_experiment.run_joint_lstm_fashionsw_mh, to_args, lstm_params, FASHION_DIR, pass_trial=False)),
    ("LSTM-Replay on Fashion-SW",make_objective(run_experiment.run_lstm_replay_fashionsw_mh, to_args, lstm_params, FASHION_DIR)),
    ("ESN-Base on Fashion-SW",   make_objective(run_experiment.run_esn_naive_fashionsw_mh, to_args, esn_base_params, FASHION_DIR)),
    ("Joint-ESN on Fashion-SW",  make_objective(run_experiment.run_joint_esn_fashionsw_mh, to_args, esn_base_params, FASHION_DIR, pass_trial=False)),
]


def run_tuning(datasets):
    STUDY_SUFFIX = {"mnist": "MNIST", "wisdm": "WISDM", "hhar": "HHAR",
                    "fashion_sw": "Fashion-SW"}
    _sel_labels  = {STUDY_SUFFIX[d] for d in datasets}
    studies = [(l, o) for (l, o) in TUNE_STUDIES
               if l.rsplit(" on ", 1)[1] in _sel_labels]

    print(f"\n{'='*70}\n  PHASE 1 — tuning {len(studies)} studies "
          f"(sequential)\n{'='*70}", flush=True)
    for _done, (label, obj) in enumerate(studies, 1):
        tuned[label] = tune(label, obj)
        print(f"  >> Phase 1 progress: {_done}/{len(studies)} studies done", flush=True)

    # Built per dataset so only selected datasets reference `tuned` keys.
    model_dataset_params = {}
    if "mnist" in datasets:
        model_dataset_params.update({
            ("gim_alstm_mh",    "mnist"):  GIM_MNIST_PARAMS,
            ("lstm_mh",         "mnist"):  tuned["LSTM on MNIST"],
            ("joint_lstm_mh",   "mnist"):  tuned["Joint-LSTM on MNIST"],
            ("lstm_replay_mh",  "mnist"):  tuned["LSTM-Replay on MNIST"],
            ("esn_naive_mh",    "mnist"):  tuned["ESN-Base on MNIST"],
            ("joint_esn_mh",    "mnist"):  tuned["Joint-ESN on MNIST"],
        })
    if "wisdm" in datasets:
        model_dataset_params.update({
            ("gim_alstm_mh",    "wisdm"):  tuned["GIM-ALSTM on WISDM"],
            ("lstm_mh",         "wisdm"):  tuned["LSTM on WISDM"],
            ("joint_lstm_mh",   "wisdm"):  tuned["Joint-LSTM on WISDM"],
            ("lstm_replay_mh",  "wisdm"):  tuned["LSTM-Replay on WISDM"],
            ("esn_naive_mh",    "wisdm"):  tuned["ESN-Base on WISDM"],
            ("joint_esn_mh",    "wisdm"):  tuned["Joint-ESN on WISDM"],
        })
    if "hhar" in datasets:
        model_dataset_params.update({
            ("gim_alstm_mh",    "hhar"):   tuned["GIM-ALSTM on HHAR"],
            ("lstm_mh",         "hhar"):   tuned["LSTM on HHAR"],
            ("joint_lstm_mh",   "hhar"):   tuned["Joint-LSTM on HHAR"],
            ("lstm_replay_mh",  "hhar"):   tuned["LSTM-Replay on HHAR"],
            ("esn_naive_mh",    "hhar"):   tuned["ESN-Base on HHAR"],
            ("joint_esn_mh",    "hhar"):   tuned["Joint-ESN on HHAR"],
        })
    if "fashion_sw" in datasets:
        model_dataset_params.update({
            # Fashion-MNIST Summer-vs-Winter (GIM is tuned here, unlike MNIST where it's fixed)
            ("gim_alstm_mh",    "fashion_sw"): tuned["GIM-ALSTM on Fashion-SW"],
            ("lstm_mh",         "fashion_sw"): tuned["LSTM on Fashion-SW"],
            ("joint_lstm_mh",   "fashion_sw"): tuned["Joint-LSTM on Fashion-SW"],
            ("lstm_replay_mh",  "fashion_sw"): tuned["LSTM-Replay on Fashion-SW"],
            ("esn_naive_mh",    "fashion_sw"): tuned["ESN-Base on Fashion-SW"],
            ("joint_esn_mh",    "fashion_sw"): tuned["Joint-ESN on Fashion-SW"],
        })
    return model_dataset_params


def get_tuned_params(datasets, cache_path=TUNED_PARAMS_CACHE):
    cached, meta = load_cache(cache_path)
    if meta.get("seed") != SEED:          # cache from a different seed -> stale
        cached, meta = {}, {}

    missing = [ds for ds in datasets if any((m, ds) not in cached for m in MODELS)]
    if missing:
        print(f"  >> tuning {missing} (not cached); other datasets reuse "
              f"{cache_path}", flush=True)
        cached.update(run_tuning(missing))
        save_cache(cache_path, cached, meta={"seed": SEED})
        print(f"  >> tuned params cached -> {cache_path}", flush=True)
    else:
        print(f"  >> all requested datasets already cached in {cache_path} "
              f"(nothing to tune)", flush=True)
    return {(m, ds): cached[(m, ds)] for ds in datasets for m in MODELS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-Head (TIL) tuning -> tuned_params.json")
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    choices=["mnist", "wisdm", "hhar", "fashion_sw"])
    cli = ap.parse_args()

    t_start = datetime.now()
    print(f"\n{'#'*70}\n#  MULTI-HEAD (TIL) TUNING\n{'#'*70}", flush=True)
    print(f"  started   : {t_start:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  device    : {DEVICE}", flush=True)
    print(f"  datasets  : {cli.datasets}", flush=True)

    get_tuned_params(cli.datasets)

    elapsed = datetime.now() - t_start
    print(f"\n{'#'*70}\n#  DONE — tuning cached\n{'#'*70}", flush=True)
    print(f"  elapsed   : {elapsed}", flush=True)
    print(f"  cache     : {TUNED_PARAMS_CACHE}", flush=True)
