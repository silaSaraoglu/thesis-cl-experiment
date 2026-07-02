"""
Single-Head (DIL) -- Phase-1 tuning, run once and cached.

This script owns ALL tuning logic. It tunes the
selected datasets and writes the best hyperparameters to tuned_params.json. run_experiment.py
and run_epoch_analysis.py only READ that cache -- they contain no tuning code -- so tuning
happens here once instead of at the start of every run.

"""
import warnings
warnings.filterwarnings("ignore")  # silence import-time lib warnings (jaxopt/Avalanche/torch)
                                   # before optuna/torch/run_experiment are imported below
import argparse
from argparse import Namespace
from datetime import datetime
import numpy as np
import torch
import optuna
import run_experiment as run_experiment
from config import (
    DATASETS, MODELS, DEVICE, SEED, EPOCHS, TUNED_PARAMS_CACHE,
    MNIST_DIR, HHAR_DIR, FASHION_DIR,
    TUNE_SUBSET_MNIST, TUNE_SUBSET_HHAR, TUNE_SUBSET_FASHION,
    TUNE_SPLIT_SEED,
)
from shared.utils import load_cache, save_cache
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 15    # trials per study

# GIM on MNIST: original paper hyperparameters (fixed, not tuned).
GIM_MNIST_PARAMS = dict(
    batch_size              = 32,
    learning_rate           = 1e-4,
    hidden_size_rnn         = 128,
    hidden_size_autoencoder = 500,
)


# One search space per model family, reused across datasets. Each (model, dataset) is still tuned independently, so the resulting best values differ per dataset.
def gim_params(trial):
    return dict(
        hidden_size_rnn         = trial.suggest_categorical("hidden_size_rnn", [32, 64, 128, 256]),
        hidden_size_autoencoder = trial.suggest_categorical("hidden_size_autoencoder", [50, 100, 200, 300]),
        learning_rate           = trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        batch_size              = trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    )

# Joint-LSTM shares the LSTM search space (lstm_params).
def lstm_params(trial):
    return dict(
        hidden_size_rnn = trial.suggest_categorical("hidden_size_rnn", [32, 64, 128, 256]),
        learning_rate   = trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        batch_size      = trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    )

def esn_base_params(trial):
    return dict(
        esn_units     = trial.suggest_categorical("esn_units", [300, 400, 500]),
        learning_rate = trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    )

# Only the strategy-specific extras are tuned here; reservoir and training params come from ESN-Base.
def esn_ewc_extra(trial):
    return dict(ewc_lambda=trial.suggest_categorical("ewc_lambda", [0.01, 0.1, 1.0, 10.0]))

def esn_lwf_extra(trial):
    return dict(
        lwf_alpha       = trial.suggest_float("lwf_alpha", 0.1, 10.0),
        lwf_temperature = trial.suggest_categorical("lwf_temperature", [0.5, 1.0, 1.5]),
    )


def to_args(extra, **kwargs):
    return Namespace(device=DEVICE, **extra, **kwargs)

tuned = {}
def make_objective(runner, arg_fn, param_fn, data_dir, base_key=None, pass_trial=True):
    """Return an Optuna objective that samples params, runs the experiment, and
    returns mean val accuracy.  base_key merges ESN-Base params for secondary studies."""
    tune_n = (TUNE_SUBSET_MNIST  if data_dir == MNIST_DIR
              else TUNE_SUBSET_HHAR if data_dir == HHAR_DIR
              else TUNE_SUBSET_FASHION)
    def objective(trial):
        # Seed every trial identically so each (params) evaluation is reproducible and
        # trials are compared on the same data shuffle / weight init. Requires tuning to
        # run serially: concurrent threads would share this global RNG and stomp the seed.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        params = param_fn(trial)
        if base_key is not None:
            params = {**tuned[base_key], **params}
        args = arg_fn({**params, "epochs": EPOCHS}, data_dir=data_dir, subset=None,
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
            n_startup_trials=min(3, n),
            n_warmup_steps=1,
        ),
    )
    study.optimize(objective, n_trials=n, show_progress_bar=False)
    print(f"  Done    {label:<40} best={study.best_value:.4f}", flush=True)
    return study.best_params


TUNE_STUDIES = [
    # GIM (MNIST uses fixed paper params; non-MNIST datasets are tuned below)
    # LSTM baselines (naive + joint)
    ("LSTM on MNIST",       make_objective(run_experiment.run_naive_lstm_mnist_sh, to_args, lstm_params, MNIST_DIR)),
    ("Joint-LSTM on MNIST", make_objective(run_experiment.run_joint_lstm_mnist_sh, to_args, lstm_params, MNIST_DIR, pass_trial=False)),
    # ESN-Base: reservoir + training tuned once per dataset via Naive strategy
    ("ESN-Base on MNIST",   make_objective(run_experiment.run_esn_naive_mnist,     to_args,  esn_base_params, MNIST_DIR)),
    # ESN strategy-specific extras (must follow Base entries)
    ("ESN-EWC on MNIST",    make_objective(run_experiment.run_esn_ewc_mnist,   to_args, esn_ewc_extra, MNIST_DIR, base_key="ESN-Base on MNIST")),
    ("ESN-LwF on MNIST",    make_objective(run_experiment.run_esn_lwf_mnist,    to_args, esn_lwf_extra,    MNIST_DIR, base_key="ESN-Base on MNIST")),
    # Joint ESN: independently tuned (all tasks seen simultaneously)
    ("Joint-ESN on MNIST",  make_objective(run_experiment.run_joint_esn_mnist_sh, to_args, esn_base_params, MNIST_DIR, pass_trial=False)),
    # HHAR studies (domain-incremental by device; single shared 6-way head)
    ("GIM-ALSTM on HHAR",    make_objective(lambda a,v,**kw: run_experiment.run_gim_hhar("alstm",a,v,**kw), to_args, gim_params, HHAR_DIR)),
    ("LSTM on HHAR",          make_objective(run_experiment.run_naive_lstm_hhar_sh,  to_args, lstm_params,       HHAR_DIR)),
    ("Joint-LSTM on HHAR",    make_objective(run_experiment.run_joint_lstm_hhar_sh,  to_args, lstm_params,       HHAR_DIR, pass_trial=False)),
    ("ESN-Base on HHAR",      make_objective(run_experiment.run_esn_naive_hhar,      to_args,  esn_base_params,   HHAR_DIR)),
    ("ESN-EWC on HHAR",       make_objective(run_experiment.run_esn_ewc_hhar,        to_args,  esn_ewc_extra,           HHAR_DIR, base_key="ESN-Base on HHAR")),
    ("ESN-LwF on HHAR",       make_objective(run_experiment.run_esn_lwf_hhar,        to_args,  esn_lwf_extra,           HHAR_DIR, base_key="ESN-Base on HHAR")),
    ("Joint-ESN on HHAR",     make_objective(run_experiment.run_joint_esn_hhar_sh,   to_args,  esn_base_params,   HHAR_DIR, pass_trial=False)),
    # Fashion-MNIST Summer-vs-Winter (GIM tuned like the other non-MNIST datasets)
    ("GIM-ALSTM on Fashion-SW", make_objective(lambda a,v,**kw: run_experiment.run_gim_fashionsw("alstm",a,v,**kw), to_args, gim_params, FASHION_DIR)),
    ("LSTM on Fashion-SW",       make_objective(run_experiment.run_naive_lstm_fashionsw_sh, to_args, lstm_params, FASHION_DIR)),
    ("Joint-LSTM on Fashion-SW", make_objective(run_experiment.run_joint_lstm_fashionsw_sh, to_args, lstm_params, FASHION_DIR, pass_trial=False)),
    ("ESN-Base on Fashion-SW",   make_objective(run_experiment.run_esn_naive_fashionsw,    to_args,  esn_base_params, FASHION_DIR)),
    ("ESN-EWC on Fashion-SW",    make_objective(run_experiment.run_esn_ewc_fashionsw,      to_args,  esn_ewc_extra,   FASHION_DIR, base_key="ESN-Base on Fashion-SW")),
    ("ESN-LwF on Fashion-SW",    make_objective(run_experiment.run_esn_lwf_fashionsw,      to_args,  esn_lwf_extra,   FASHION_DIR, base_key="ESN-Base on Fashion-SW")),
    ("Joint-ESN on Fashion-SW",  make_objective(run_experiment.run_joint_esn_fashionsw_sh, to_args,  esn_base_params, FASHION_DIR, pass_trial=False)),
]


def run_tuning(datasets):
    STUDY_SUFFIX = {"mnist": "MNIST", "hhar": "HHAR", "fashion_sw": "Fashion-SW"}
    _sel_labels  = {STUDY_SUFFIX[d] for d in datasets}
    studies = [(l, o) for (l, o) in TUNE_STUDIES
               if l.rsplit(" on ", 1)[1] in _sel_labels]

    # Phase 1a: independent studies (LSTM, Joint-LSTM, ESN-Base, Joint-ESN, GIM)
    # Phase 1b: ESN secondary studies (EWC/LwF) must follow their ESN-Base
    SECONDARY = ("ESN-EWC", "ESN-LwF")
    primary_studies   = [(l, o) for l, o in studies if not l.startswith(SECONDARY)]
    secondary_studies = [(l, o) for l, o in studies if l.startswith(SECONDARY)]

    print(f"\n{'='*70}\n  PHASE 1a — tuning {len(primary_studies)} primary studies "
          f"(serial)\n{'='*70}", flush=True)
    for _done, (label, obj) in enumerate(primary_studies, 1):
        tuned[label] = tune(label, obj)
        print(f"  >> Phase 1a progress: {_done}/{len(primary_studies)} studies done", flush=True)

    print(f"\n{'='*70}\n  PHASE 1b — tuning {len(secondary_studies)} ESN strategy studies "
          f"(serial)\n{'='*70}", flush=True)
    for _done, (label, obj) in enumerate(secondary_studies, 1):
        ds = label.split(" on ", 1)[1]
        tuned[label] = {**tuned[f"ESN-Base on {ds}"], **tune(label, obj)}
        print(f"  >> Phase 1b progress: {_done}/{len(secondary_studies)} studies done", flush=True)

    # Built per dataset so only selected datasets reference `tuned` keys.
    model_dataset_params = {}
    if "mnist" in datasets:
        model_dataset_params.update({
            ("gim_alstm_sh",    "mnist"):   GIM_MNIST_PARAMS,
            ("lstm_sh",         "mnist"):   tuned["LSTM on MNIST"],
            ("lstm_replay_sh",  "mnist"):   tuned["LSTM on MNIST"],
            ("joint_lstm_sh",   "mnist"):   tuned["Joint-LSTM on MNIST"],
            # ESN: naive and slda share base reservoir params; ewc/lwf/replay carry merged dicts
            ("esn_naive_sh",    "mnist"):   tuned["ESN-Base on MNIST"],
            ("esn_ewc_sh",      "mnist"):   tuned["ESN-EWC on MNIST"],
            ("esn_lwf_sh",      "mnist"):   tuned["ESN-LwF on MNIST"],
            ("esn_replay_sh",   "mnist"):   tuned["ESN-Base on MNIST"],
            ("esn_slda_sh",     "mnist"):   tuned["ESN-Base on MNIST"],
            ("joint_esn_sh",    "mnist"):   tuned["Joint-ESN on MNIST"],
        })
    if "hhar" in datasets:
        model_dataset_params.update({
            ("gim_alstm_sh",    "hhar"):    tuned["GIM-ALSTM on HHAR"],
            ("lstm_sh",         "hhar"):    tuned["LSTM on HHAR"],
            ("lstm_replay_sh",  "hhar"):    tuned["LSTM on HHAR"],
            ("joint_lstm_sh",   "hhar"):    tuned["Joint-LSTM on HHAR"],
            ("esn_naive_sh",    "hhar"):    tuned["ESN-Base on HHAR"],
            ("esn_ewc_sh",      "hhar"):    tuned["ESN-EWC on HHAR"],
            ("esn_lwf_sh",      "hhar"):    tuned["ESN-LwF on HHAR"],
            ("esn_replay_sh",   "hhar"):    tuned["ESN-Base on HHAR"],
            ("esn_slda_sh",     "hhar"):    tuned["ESN-Base on HHAR"],
            ("joint_esn_sh",    "hhar"):    tuned["Joint-ESN on HHAR"],
        })
    if "fashion_sw" in datasets:
        model_dataset_params.update({
            # Fashion-MNIST Summer-vs-Winter (GIM tuned like the other non-MNIST datasets)
            ("gim_alstm_sh",    "fashion_sw"): tuned["GIM-ALSTM on Fashion-SW"],
            ("lstm_sh",         "fashion_sw"): tuned["LSTM on Fashion-SW"],
            ("lstm_replay_sh",  "fashion_sw"): tuned["LSTM on Fashion-SW"],
            ("joint_lstm_sh",   "fashion_sw"): tuned["Joint-LSTM on Fashion-SW"],
            ("esn_naive_sh",    "fashion_sw"): tuned["ESN-Base on Fashion-SW"],
            ("esn_ewc_sh",      "fashion_sw"): tuned["ESN-EWC on Fashion-SW"],
            ("esn_lwf_sh",      "fashion_sw"): tuned["ESN-LwF on Fashion-SW"],
            ("esn_replay_sh",   "fashion_sw"): tuned["ESN-Base on Fashion-SW"],
            ("esn_slda_sh",     "fashion_sw"): tuned["ESN-Base on Fashion-SW"],
            ("joint_esn_sh",    "fashion_sw"): tuned["Joint-ESN on Fashion-SW"],
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
    ap = argparse.ArgumentParser(description="Single-Head (DIL) tuning -> tuned_params.json")
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    choices=["mnist", "hhar", "fashion_sw"])
    cli = ap.parse_args()

    t_start = datetime.now()
    print(f"\n{'#'*70}\n#  SINGLE-HEAD (DIL) TUNING\n{'#'*70}", flush=True)
    print(f"  started   : {t_start:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  device    : {DEVICE}", flush=True)
    print(f"  datasets  : {cli.datasets}", flush=True)

    get_tuned_params(cli.datasets)

    elapsed = datetime.now() - t_start
    print(f"\n{'#'*70}\n#  DONE — tuning cached\n{'#'*70}", flush=True)
    print(f"  elapsed   : {elapsed}", flush=True)
    print(f"  cache     : {TUNED_PARAMS_CACHE}", flush=True)
