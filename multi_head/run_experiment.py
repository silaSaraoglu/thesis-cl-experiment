"""
Multi-Head (TIL Setting 2) Continual Learning Pipeline.

Architecture: per-task linear heads via ModuleDict.  Oracle task ID provided
at test time — this is the Task-Incremental Learning (TIL) upper bound.

Phase 1 -- Hyperparameter tuning (same search spaces as SH):
  - GIM-ALSTM / GIM-ALMN  : HAR
  - LSTM / Joint-LSTM      : MNIST, HAR
  - ESN-Base (Naive MH)    : MNIST, HAR
  - Joint-ESN              : MNIST, HAR

Phase 2 -- Full experiment with best found parameters.

Usage:
    python run_experiment.py
"""
import sys
try:  # force UTF-8 console so Unicode in summaries never crashes on Windows (cp1254)
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import os

os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import optuna
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import Namespace
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
_p = HERE
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths
sys.path.insert(0, HERE)  # multi_head/ must beat single_head/ for 'models.*' lookups

from models.GIM_LSTM.gim_lstm_mnist_mh import run_gim_mnist_mh
from models.GIM_LSTM.gim_lstm_wisdm_mh import run_gim_wisdm_mh
from models.LSTM.naive.naive_lstm_mnist_mh import run_naive_lstm_mnist_mh
from models.LSTM.naive.naive_lstm_wisdm_mh import run_naive_lstm_wisdm_mh
from models.LSTM.joint.joint_lstm_mnist_mh import run_joint_lstm_mnist_mh
from models.LSTM.joint.joint_lstm_wisdm_mh import run_joint_lstm_wisdm_mh
from models.LSTM.replay.replay_lstm_mnist_mh import run_lstm_replay_mnist_mh
from models.LSTM.replay.replay_lstm_wisdm_mh import run_lstm_replay_wisdm_mh
from models.ESN.naive.naive_esn_mnist_mh import run_esn_naive_mnist_mh
from models.ESN.naive.naive_esn_wisdm_mh import run_esn_naive_wisdm_mh
from models.ESN.joint.joint_esn_mnist_mh import run_joint_esn_mnist_mh
from models.ESN.joint.joint_esn_wisdm_mh import run_joint_esn_wisdm_mh
# HHAR (TIL by device; per-device 6-way heads)
from models.GIM_LSTM.gim_lstm_hhar_mh import run_gim_hhar_mh
from models.LSTM.naive.naive_lstm_hhar_mh import run_naive_lstm_hhar_mh
from models.LSTM.joint.joint_lstm_hhar_mh import run_joint_lstm_hhar_mh
from models.LSTM.replay.replay_lstm_hhar_mh import run_lstm_replay_hhar_mh
from models.ESN.naive.naive_esn_hhar_mh import run_esn_naive_hhar_mh
from models.ESN.joint.joint_esn_hhar_mh import run_joint_esn_hhar_mh
from models.GIM_LSTM.gim_lstm_fashion_mh import run_gim_fashion_mh
from models.LSTM.naive.naive_lstm_fashion_mh import run_naive_lstm_fashion_mh
from models.LSTM.joint.joint_lstm_fashion_mh import run_joint_lstm_fashion_mh
from models.LSTM.replay.replay_lstm_fashion_mh import run_lstm_replay_fashion_mh
from models.ESN.naive.naive_esn_fashion_mh import run_esn_naive_fashion_mh
from models.ESN.joint.joint_esn_fashion_mh import run_joint_esn_fashion_mh
from models.GIM_LSTM.gim_lstm_fashionsw_mh import run_gim_fashionsw_mh
from models.LSTM.naive.naive_lstm_fashionsw_mh import run_naive_lstm_fashionsw_mh
from models.LSTM.joint.joint_lstm_fashionsw_mh import run_joint_lstm_fashionsw_mh
from models.LSTM.replay.replay_lstm_fashionsw_mh import run_lstm_replay_fashionsw_mh
from models.ESN.naive.naive_esn_fashionsw_mh import run_esn_naive_fashionsw_mh
from models.ESN.joint.joint_esn_fashionsw_mh import run_joint_esn_fashionsw_mh
from shared.metrics import save_results

# Global settings
DATASETS    = ["mnist", "wisdm", "hhar", "fashion", "fashion_sw"]
MODELS      = ["gim_alstm_mh", "lstm_mh", "joint_lstm_mh",
                "lstm_replay_mh",
                "esn_naive_mh", "joint_esn_mh"]
NUM_RUNS    = 1
SEED        = 42
RESULTS_DIR = os.path.join(HERE, "..", "results", "multi_head")
MNIST_DIR   = os.path.join(HERE, "..", "data", "mnist")
WISDM_DIR     = os.path.join(HERE, "..", "data", "wisdm")
HHAR_DIR      = os.path.join(HERE, "..", "data", "hhar")
FASHION_DIR   = os.path.join(HERE, "..", "data", "fashion_mnist")
SUBSET_MNIST      = None   # samples per task for Phase 2 (None = full dataset)
SUBSET_WISDM      = None   # samples per task for Phase 2 (None = full dataset)
SUBSET_HHAR       = None   # samples per task for Phase 2 (None = full dataset)
SUBSET_FASHION    = None   # samples per task for Phase 2 (None = full dataset)

# MNIST task-pair configurations
# Three digit-pairing variants; results are averaged across all three.
MNIST_DATASET_CONFIGS = {
    "config_1": [[0,1],[2,3],[4,5],[6,7],[8,9]],  # 0v1, 2v3, 4v5, 6v7, 8v9
    "config_2": [[1,2],[3,4],[5,6],[7,8],[0,9]],  # 1v2, 3v4, 5v6, 7v8, 0v9
    "config_3": [[0,3],[2,5],[4,7],[6,9],[8,1]],  # 0v3, 2v5, 4v7, 6v9, 8v1 (mixed-parity, matches SH)
}

# WISDM task-ordering configurations
# Same 3 activity pairs, three different presentation orders; averaged.
WISDM_DATASET_CONFIGS = {
    "config_1": [[0,1],[2,3],[4,5]],  # Walk/Jog · Up/Down · Sit/Stand
    "config_2": [[0,1],[4,5],[2,3]],  # Walk/Jog · Sit/Stand · Up/Down
    "config_3": [[2,3],[0,1],[4,5]],  # Up/Down  · Walk/Jog · Sit/Stand
}

# HHAR task orderings (device-model indices: 0=nexus4, 1=s3, 2=s3mini, 3=samsungold).
HHAR_DATASET_CONFIGS = {
    "config_1": [0, 1, 2, 3],   # nexus4 -> s3 -> s3mini -> samsungold
    "config_2": [3, 2, 1, 0],   # reverse
    "config_3": [0, 2, 1, 3],   # alternating cross-cluster (max domain jumps)
}

# Datasets that use multiple configs

# Fashion-MNIST garment-vs-accessory task orderings (pairs are (garment, accessory)).
FASHION_DATASET_CONFIGS = {
    "config_1": [[0,5],[1,7],[2,9],[6,8]],
    "config_2": [[6,8],[2,9],[1,7],[0,5]],
    "config_3": [[1,7],[6,8],[0,5],[2,9]],
}


# Fashion-MNIST Summer-vs-Winter task orderings (within-category season pairs).
FASHION_SW_DATASET_CONFIGS = {
    "config_1": [[0,2],[3,4],[5,9]],
    "config_2": [[5,9],[3,4],[0,2]],
    "config_3": [[3,4],[5,9],[0,2]],
}

DATASET_CONFIGS = {"mnist": MNIST_DATASET_CONFIGS, "wisdm": WISDM_DATASET_CONFIGS, "hhar": HHAR_DATASET_CONFIGS, "fashion": FASHION_DATASET_CONFIGS, "fashion_sw": FASHION_SW_DATASET_CONFIGS}


def ordering_label(pairs, kind):
    """Readable task ordering for a config, e.g. 'G3/G4 > G5/G6 > G1/G2 > G7/G8'."""
    wisdm_names = {(0, 1): "Walk/Jog", (2, 3): "Up/Down", (4, 5): "Sit/Stand"}
    parts = []
    for a, b in pairs:
        if kind == "wisdm":
            parts.append(wisdm_names.get((a, b), f"{a}/{b}"))
        else:
            parts.append(f"{a}/{b}")
    return " > ".join(parts)


def average_r_matrices(r_matrices):
    """Average serialized R-matrices across task configurations."""
    cells = {}
    for R_raw in r_matrices:
        for i_s, row in R_raw.items():
            for j_s, val in row.items():
                cells.setdefault((str(i_s), str(j_s)), []).append(float(val))
    averaged = {}
    for (i_s, j_s), vals in cells.items():
        averaged.setdefault(i_s, {})[j_s] = float(np.mean(vals))
    return averaged


DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import threading
PRINT_LOCK        = threading.Lock()

N_TRIALS          = 15
N_STRATEGY_TRIALS = 5
# Samples per task reserved as the tuning holdout pool (~10% of each task's training data).
TUNE_SUBSET_MNIST = 1200   # ~10% of 12,000 per digit pair
TUNE_SUBSET_WISDM = 62     # ~10% of 622 per activity pair (balanced)
TUNE_SUBSET_HHAR  = 600    # ~10% of ~6750 windows per device (held out for tuning)
TUNE_SUBSET_FASHION = 1200  # ~10% of 12,000 per garment/accessory pair
TUNE_SPLIT_SEED   = 0      # fixed seed for tuning/training data split

# GIM on MNIST: original paper hyperparameters
GIM_MNIST_PARAMS = dict(
    epochs                  = 2,
    batch_size              = 32,
    learning_rate           = 1e-4,
    hidden_size_rnn         = 128,
    hidden_size_autoencoder = 500,
)

# Optuna param samplers
# One search space per model family, reused for every dataset (MNIST / WISDM / HHAR).
# Each (model, dataset) study is tuned independently, so the saved best values differ.
def gim_params(trial):
    return dict(
        hidden_size_rnn         = trial.suggest_categorical("hidden_size_rnn", [64, 128, 256]),
        hidden_size_autoencoder = trial.suggest_categorical("hidden_size_autoencoder", [500, 600, 700]),
        learning_rate           = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        batch_size              = trial.suggest_categorical("batch_size", [32, 64]),
        epochs                  = trial.suggest_int("epochs", 5, 15),
    )

# Joint-LSTM shares the LSTM search space.
def lstm_params(trial):
    return dict(
        hidden_size_rnn = trial.suggest_categorical("hidden_size_rnn", [64, 128, 256]),
        learning_rate   = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        batch_size      = trial.suggest_categorical("batch_size", [32, 64]),
        epochs          = trial.suggest_int("epochs", 5, 15),
    )

def esn_base_params(trial):
    return dict(
        esn_units     = trial.suggest_categorical("esn_units", [500, 1000, 2000]),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        batch_size    = trial.suggest_categorical("batch_size", [32, 64]),
        epochs        = trial.suggest_int("epochs", 5, 15),
    )

# Arg-builder + objective factory for tuning
def to_args(extra, **kwargs):
    return Namespace(device=DEVICE, **extra, **kwargs)

tuned = {}
def make_objective(runner, arg_fn, param_fn, data_dir, base_key=None, pass_trial=True):
    """Return an Optuna objective that samples params, runs the experiment, and
    returns mean val accuracy.  base_key merges ESN-Base params for secondary studies."""
    tune_n = (TUNE_SUBSET_MNIST  if data_dir == MNIST_DIR
              else TUNE_SUBSET_WISDM if data_dir == WISDM_DIR
              else TUNE_SUBSET_HHAR if data_dir == HHAR_DIR
              else TUNE_SUBSET_FASHION)
    def objective(trial):
        params = param_fn(trial)
        if base_key is not None:
            params = {**tuned[base_key], **params}
        args = arg_fn(params, data_dir=data_dir, subset=None,
                      holdout_n=tune_n, holdout_seed=TUNE_SPLIT_SEED, use_holdout=True)
        trial_kwargs = {"trial": trial} if pass_trial else {}
        return float(np.mean(runner(args, False, **trial_kwargs)[0]))
    return objective

# Tuner
def tune(label, objective, n_trials = None):
    n = n_trials if n_trials is not None else N_TRIALS
    with PRINT_LOCK:
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
    with PRINT_LOCK:
        print(f"  Done    {label:<40} best={study.best_value:.4f}", flush=True)
    return study.best_params

# Task metadata & runners
TASK_META = {
    "mnist":   {"num_tasks": 5, "task_names": {0:"0/1",1:"2/3",2:"4/5",3:"6/7",4:"8/9"}},
    "wisdm":     {"num_tasks": 3, "task_names": {0:"Walk/Jog",1:"Up/Down",2:"Sit/Stand"}},
    "hhar":      {"num_tasks": 4, "task_names": {0:"nexus4",1:"s3",2:"s3mini",3:"samsungold"}},
    "fashion":   {"num_tasks": 4, "task_names": {0:"Tshirt/Sandal",1:"Trouser/Sneaker",2:"Pullover/Boot",3:"Shirt/Bag"}},
    "fashion_sw":{"num_tasks": 3, "task_names": {0:"Tshirt/Pullover",1:"Dress/Coat",2:"Sandal/Ankle-boot"}},
}

DATASET_LABEL = {"mnist": "MNIST", "wisdm": "WISDM", "hhar": "HHAR", "fashion": "Fashion-GA", "fashion_sw": "Fashion-SW"}

RUNNERS = {
    ("mnist", "gim_alstm_mh"):    lambda a, v: run_gim_mnist_mh("alstm", a, v),

    ("mnist", "lstm_mh"):         lambda a, v: run_naive_lstm_mnist_mh(a, v),
    ("mnist", "joint_lstm_mh"):   lambda a, v: run_joint_lstm_mnist_mh(a, v),
    ("mnist", "lstm_replay_mh"):  lambda a, v: run_lstm_replay_mnist_mh(a, v),
    ("mnist", "esn_naive_mh"):    lambda a, v: run_esn_naive_mnist_mh(a, v),
    ("mnist", "joint_esn_mh"):    lambda a, v: run_joint_esn_mnist_mh(a, v),
    ("wisdm",   "gim_alstm_mh"):    lambda a, v: run_gim_wisdm_mh("alstm", a, v),

    ("wisdm",   "lstm_mh"):         lambda a, v: run_naive_lstm_wisdm_mh(a, v),
    ("wisdm",   "joint_lstm_mh"):   lambda a, v: run_joint_lstm_wisdm_mh(a, v),
    ("wisdm",   "lstm_replay_mh"):  lambda a, v: run_lstm_replay_wisdm_mh(a, v),
    ("wisdm",   "esn_naive_mh"):    lambda a, v: run_esn_naive_wisdm_mh(a, v),
    ("wisdm",   "joint_esn_mh"):    lambda a, v: run_joint_esn_wisdm_mh(a, v),

    ("hhar",    "gim_alstm_mh"):    lambda a, v: run_gim_hhar_mh("alstm", a, v),
    ("hhar",    "lstm_mh"):         lambda a, v: run_naive_lstm_hhar_mh(a, v),
    ("hhar",    "joint_lstm_mh"):   lambda a, v: run_joint_lstm_hhar_mh(a, v),
    ("hhar",    "lstm_replay_mh"):  lambda a, v: run_lstm_replay_hhar_mh(a, v),
    ("hhar",    "esn_naive_mh"):    lambda a, v: run_esn_naive_hhar_mh(a, v),
    ("hhar",    "joint_esn_mh"):    lambda a, v: run_joint_esn_hhar_mh(a, v),
    ("fashion", "gim_alstm_mh"):    lambda a, v: run_gim_fashion_mh("alstm", a, v),
    ("fashion", "lstm_mh"):         lambda a, v: run_naive_lstm_fashion_mh(a, v),
    ("fashion", "joint_lstm_mh"):   lambda a, v: run_joint_lstm_fashion_mh(a, v),
    ("fashion", "lstm_replay_mh"):  lambda a, v: run_lstm_replay_fashion_mh(a, v),
    ("fashion", "esn_naive_mh"):    lambda a, v: run_esn_naive_fashion_mh(a, v),
    ("fashion", "joint_esn_mh"):    lambda a, v: run_joint_esn_fashion_mh(a, v),
    ("fashion_sw","gim_alstm_mh"):    lambda a, v: run_gim_fashionsw_mh("alstm", a, v),
    ("fashion_sw","lstm_mh"):         lambda a, v: run_naive_lstm_fashionsw_mh(a, v),
    ("fashion_sw","joint_lstm_mh"):   lambda a, v: run_joint_lstm_fashionsw_mh(a, v),
    ("fashion_sw","lstm_replay_mh"):  lambda a, v: run_lstm_replay_fashionsw_mh(a, v),
    ("fashion_sw","esn_naive_mh"):    lambda a, v: run_esn_naive_fashionsw_mh(a, v),
    ("fashion_sw","joint_esn_mh"):    lambda a, v: run_joint_esn_fashionsw_mh(a, v),
}

SKIP_KEYS = {"results_dir", "subset", "data_dir", "device",
             "holdout_n", "holdout_seed", "use_holdout"}

def _build_args(model, dataset, model_dataset_params):
    args = Namespace(**model_dataset_params[(model, dataset)])
    args.results_dir = RESULTS_DIR
    if dataset == "mnist":
        args.data_dir    = MNIST_DIR
        args.subset      = SUBSET_MNIST
        args.holdout_n   = TUNE_SUBSET_MNIST
    elif dataset == "wisdm":
        args.data_dir    = WISDM_DIR
        args.subset      = SUBSET_WISDM
        args.holdout_n   = TUNE_SUBSET_WISDM
    elif dataset == "hhar":
        args.data_dir    = HHAR_DIR
        args.subset      = SUBSET_HHAR
        args.holdout_n   = TUNE_SUBSET_HHAR
    elif dataset == "fashion":
        args.data_dir    = FASHION_DIR
        args.subset      = SUBSET_FASHION
        args.holdout_n   = TUNE_SUBSET_FASHION
    elif dataset == "fashion_sw":
        args.data_dir    = FASHION_DIR
        args.subset      = SUBSET_FASHION
        args.holdout_n   = TUNE_SUBSET_FASHION
    args.holdout_seed = TUNE_SPLIT_SEED
    args.use_holdout  = False
    args.device       = DEVICE
    return args

def run_combination(dataset, model, model_dataset_params, exp_dir):
    meta          = TASK_META[dataset]
    num_tasks     = meta["num_tasks"]
    names         = meta["task_names"]
    runner        = RUNNERS[(dataset, model)]
    args          = _build_args(model, dataset, model_dataset_params)
    model_label   = model.upper().replace("_", "-")
    dataset_label = DATASET_LABEL[dataset]

    with PRINT_LOCK:
        print(f"  Running {model_label:<15} on {dataset_label:<12} ...", flush=True)

    hyperparams = {k: v for k, v in vars(args).items() if k not in SKIP_KEYS}

    task_cfgs = DATASET_CONFIGS.get(dataset)
    if task_cfgs is not None:
        per_config = {}
        for cfg_name, task_pairs in task_cfgs.items():
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            run_args = Namespace(**vars(args), task_pairs=task_pairs)
            result   = runner(run_args, False)
            per_config[cfg_name] = {
                "val":      [float(v) for v in result[0]],
                "test":     [float(v) for v in result[1]],
                "summary":  result[2].summary(),
                "R_matrix": {str(i): {str(j): v for j, v in d.items()}
                             for i, d in result[2].R.items()},
                "task_pairs": task_pairs,
            }
        all_summaries = [pc["summary"] for pc in per_config.values()]
        _skeys = list(all_summaries[0].keys())
        summary = {k: (float(np.mean([s[k] for s in all_summaries if s[k] is not None]))
                       if any(s[k] is not None for s in all_summaries) else None)
                   for k in _skeys}
        val_accs  = [float(np.mean([per_config[c]["val"][t]  for c in per_config])) for t in range(num_tasks)]
        test_accs = [float(np.mean([per_config[c]["test"][t] for c in per_config])) for t in range(num_tasks)]
        last_R = per_config[next(iter(per_config))]["R_matrix"]
    else:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        run_args = args
        result   = runner(run_args, False)
        val_accs  = [float(v) for v in result[0]]
        test_accs = [float(v) for v in result[1]]
        summary   = result[2].summary()
        last_R    = {str(i): {str(j): v for j, v in d.items()} for i, d in result[2].R.items()}
        per_config = None

    out = {
        "model":       f"{model_label}-{dataset_label}",
        "dataset":     dataset,
        "model_id":    model,
        "hyperparams": hyperparams,
        "summary":     summary,
        "table": {
            names.get(t, str(t+1)): {"val": val_accs[t], "test": test_accs[t]}
            for t in range(num_tasks)
        },
        "R_matrix":   last_R,
        "per_config": per_config,
    }
    save_results(out, os.path.join(exp_dir, f"{model}_{dataset}.json"))

    return {
        "model":       model_label,
        "dataset":     dataset_label,
        "hyperparams": hyperparams,
        "summary":     summary,
        "val_mean":    val_accs,
        "test_mean":   test_accs,
        "task_names":  [names.get(t, str(t+1)) for t in range(num_tasks)],
        "R_matrix":    last_R,
        "per_config":  per_config,
    }

def format_metric(v):
    """Format a metric value; show N/A for None (joint-training runs only report final_accuracy and final_cohen_kappa)."""
    return "  N/A " if v is None else f"{v:.4f}"

def print_hyperparams_table(all_results):
    print(f"\n{'='*70}")
    print("  BEST HYPERPARAMETERS")
    print(f"{'='*70}")
    for r in all_results:
        print(f"\n  {r['model']} on {r['dataset']}")
        print(f"  {'-'*40}")
        for k, v in sorted(r["hyperparams"].items()):
            print(f"    {k:<28} = {v}")

def print_metrics_table(all_results):
    cols = ["Model", "Dataset", "Acc_final", "Acc_avg", "BWT_final", "BWT_avg",
            "FWT", "Kappa", "Plasticity", "Stability"]
    col_widths = [15, 12, 10, 10, 10, 10, 8, 8, 11, 10]
    sep  = "  " + "  ".join("-" * c for c in col_widths)

    print(f"\n{'='*110}")
    print("  CL METRICS SUMMARY")
    print(f"{'='*110}")
    header = "  " + "  ".join(f"{c:<{col_widths[i]}}" for i, c in enumerate(cols))
    print(header)
    print(sep)
    for r in all_results:
        summary = r["summary"]
        row = [
            r["model"], r["dataset"],
            format_metric(summary.get('final_accuracy')),             format_metric(summary.get('average_accuracy_over_time')),
            format_metric(summary.get('final_backward_transfer')),   format_metric(summary.get('average_backward_transfer')),
            format_metric(summary.get('forward_transfer')),           format_metric(summary.get('final_cohen_kappa')),
            format_metric(summary.get('plasticity')),                 format_metric(summary.get('stability')),
        ]
        print("  " + "  ".join(f"{v:<{col_widths[i]}}" for i, v in enumerate(row)))
    print(sep)

    print(f"\n{'='*90}")
    print("  PER-TASK TEST ACCURACY")
    print(f"{'='*90}")
    for r in all_results:
        task_str = "  ".join(
            f"{name}: {acc:.4f}"
            for name, acc in zip(r["task_names"], r["test_mean"])
        )
        print(f"  {r['model']:<15} {r['dataset']:<12}  {task_str}")

def save_plots(all_results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})

    def _fig_footer(fig):
        g = globals()
        parts = []
        sm, sw = g.get("SUBSET_MNIST"), g.get("SUBSET_WISDM")
        if sm is not None or sw is not None:
            parts.append(
                f"Train subset: MNIST={sm if sm is not None else 'full'}, "
                f"WISDM={sw if sw is not None else 'full'}"
            )
        tm, tw = g.get("TUNE_SUBSET_MNIST"), g.get("TUNE_SUBSET_WISDM")
        if tm is not None:
            parts.append(f"Tune subset: MNIST={tm}, WISDM={tw}")
        if parts:
            fig.text(0.5, 0.002, "  |  ".join(parts),
                     ha="center", va="bottom", fontsize=7,
                     color="#777777", style="italic")

    plots_dir  = os.path.join(results_dir, "plots")
    rmat_dir   = os.path.join(plots_dir, "rmatrix")
    os.makedirs(rmat_dir, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in all_results))

    for ds in datasets:
        rs     = [r for r in all_results if r["dataset"] == ds]
        labels = [r["model"] for r in rs]
        x      = np.arange(len(rs))
        slug   = ds.lower().replace("-", "").replace(" ", "")

        # 1. CL metrics overview (2×4) with inline definitions
        metric_keys = [
            "final_accuracy", "average_accuracy_over_time",
            "final_backward_transfer", "average_backward_transfer",
            "forward_transfer", "final_cohen_kappa",
            "plasticity", "stability",
        ]
        metric_titles = [
            ("Final Accuracy",         "mean R[T−1][j] over all j"),
            ("Avg Accuracy Over Time", "mean R[i][j] for all i≥j (lower triangle)"),
            ("Final BWT",              "mean(R[T−1][j]−R[j][j]) for j<T−1"),
            ("Avg BWT",                "mean(R[i][j]−R[j][j]) for all i>j"),
            ("Forward Transfer",       "mean(R[j−1][j]−0.5) for j>0"),
            ("Cohen's Kappa",          "κ=(p_o−p_e)/(1−p_e), agreement above chance"),
            ("Plasticity",             "mean diagonal R[j][j]: learning of current task"),
            ("Stability",              "mean off-diag (i>j): retention of past tasks"),
        ]

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f"CL Metrics (MH) — {ds}", fontsize=12, fontweight="bold")

        for ax, key, (title, defn) in zip(axes.flat, metric_keys, metric_titles):
            # Only models that DEFINE this metric get a bar/slot (e.g. joint models
            # report no BWT/FWT/avg-acc/plasticity/stability, so they are dropped here).
            defined_models = [(lab, r["summary"].get(key),
                               r.get("summary_std", {}).get(key) or 0.0,
                               COLORS[i % len(COLORS)])
                              for i, (lab, r) in enumerate(zip(labels, rs))
                              if r["summary"].get(key) is not None]
            ax.set_title(f"{title}\n[{defn}]", fontsize=8, pad=4)
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            if not defined_models:
                ax.set_xticks([]); ax.set_ylim(-0.05, 0.05)
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#999999")
                continue
            sub_labels = [d[0] for d in defined_models]
            sub_vals   = [d[1] for d in defined_models]
            sub_errs   = [d[2] for d in defined_models]
            sub_colors = [d[3] for d in defined_models]
            xs = np.arange(len(defined_models))
            bars = ax.bar(xs, sub_vals, color=sub_colors, edgecolor="white", linewidth=0.5)
            ax.errorbar(xs, sub_vals, yerr=sub_errs, fmt="none", color="black",
                        capsize=3, linewidth=1, zorder=6)
            ax.set_xticks(xs)
            ax.set_xticklabels(sub_labels, rotation=40, ha="right", fontsize=7)
            ymin = min(min(sub_vals) - 0.05, -0.05)
            ymax = max(max(sub_vals) + 0.05,  0.05)
            ax.set_ylim(ymin, ymax)
            if key in ("final_backward_transfer", "average_backward_transfer",
                       "forward_transfer"):
                ax.axhspan(0,    ymax, alpha=0.07, color="green", zorder=0)
                ax.axhspan(ymin, 0,   alpha=0.07, color="red",   zorder=0)
            for bar, v in zip(bars, sub_vals):
                va  = "bottom" if v >= 0 else "top"
                off = 0.01    if v >= 0 else -0.02
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off,
                        f"{v:.2f}", ha="center", va=va, fontsize=6)

        plt.tight_layout()
        _fig_footer(fig)
        fig.savefig(os.path.join(plots_dir, f"metrics_{slug}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. Per-task final test accuracy
        task_names = rs[0]["task_names"]
        n_tasks    = len(task_names)
        xt         = np.arange(n_tasks)
        width      = 0.8 / len(rs)

        fig, ax = plt.subplots(figsize=(max(8, n_tasks * len(rs) * 0.35 + 2), 5))
        for i, r in enumerate(rs):
            offset = (i - len(rs) / 2 + 0.5) * width
            ax.bar(xt + offset, r["test_mean"], width,
                   label=r["model"], color=COLORS[i % len(COLORS)],
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(xt)
        ax.set_xticklabels(task_names, fontsize=8)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(
            f"Per-Task Final Test Accuracy (MH) — {ds}\n"
            "[accuracy on each task after training ALL tasks — R[T−1][j]]",
            fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", label="chance")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        plt.tight_layout()
        _fig_footer(fig)
        fig.savefig(os.path.join(plots_dir, f"per_task_{slug}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # sequential models only (have >1 row in R-matrix)
        seq_rs = [r for r in rs if r.get("R_matrix") and len(r["R_matrix"]) > 1]

        # 5. Per-task average accuracy
        # avg_acc_j = mean(R[i][j] for i >= j): average retention of task j over time
        if seq_rs:
            width5 = 0.8 / len(seq_rs)
            fig, ax = plt.subplots(
                figsize=(max(8, n_tasks * len(seq_rs) * 0.35 + 2), 5))
            ax.set_title(
                f"Per-Task Average Accuracy (MH) — {ds}\n"
                "[avg_acc_j = mean(R[i][j] for i≥j): how well each task is retained on average]",
                fontweight="bold")
            for i, r in enumerate(seq_rs):
                R_raw = r["R_matrix"]
                T     = max(int(k) for k in R_raw) + 1
                avgs  = []
                for j in range(T):
                    col = [R_raw.get(str(ii), {}).get(str(j))
                           for ii in range(j, T)]
                    col = [v for v in col if v is not None]
                    avgs.append(np.mean(col) if col else 0.0)
                offset = (i - len(seq_rs) / 2 + 0.5) * width5
                ax.bar(np.arange(T) + offset, avgs, width5,
                       label=r["model"], color=COLORS[i % len(COLORS)],
                       edgecolor="white", linewidth=0.4)
            ax.set_xticks(np.arange(n_tasks))
            ax.set_xticklabels(task_names, fontsize=8)
            ax.set_ylabel("Average Accuracy")
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", label="chance")
            ax.legend(fontsize=7, ncol=3, loc="upper right")
            plt.tight_layout()
            _fig_footer(fig)
            fig.savefig(os.path.join(plots_dir, f"per_task_avg_acc_{slug}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 6. Per-task BWT
        # BWT_j = R[T−1][j] − R[j][j] for j = 0..T−2
        # negative = forgetting, positive = improvement after training later tasks
        if seq_rs and n_tasks > 1:
            width6 = 0.8 / len(seq_rs)
            xt_bwt = np.arange(n_tasks - 1)
            all_bwt_vals = []
            bwt_per_model = []
            for r in seq_rs:
                R_raw = r["R_matrix"]
                T     = max(int(k) for k in R_raw) + 1
                bwts  = []
                for j in range(T - 1):
                    rTj = R_raw.get(str(T - 1), {}).get(str(j))
                    rjj = R_raw.get(str(j),     {}).get(str(j))
                    bwts.append((rTj - rjj) if (rTj is not None and rjj is not None) else 0.0)
                bwt_per_model.append(bwts)
                all_bwt_vals.extend(bwts)
            ymin_b = min(min(all_bwt_vals) - 0.05, -0.05)
            ymax_b = max(max(all_bwt_vals) + 0.05,  0.05)

            fig, ax = plt.subplots(
                figsize=(max(8, (n_tasks - 1) * len(seq_rs) * 0.35 + 2), 5))
            ax.set_title(
                f"Per-Task Backward Transfer (MH) — {ds}\n"
                "[BWT_j = R[T−1][j] − R[j][j] | red = forgetting, green = improvement]",
                fontweight="bold")
            ax.axhspan(0,      ymax_b, alpha=0.07, color="green", zorder=0)
            ax.axhspan(ymin_b, 0,      alpha=0.07, color="red",   zorder=0)
            for i, (r, bwts) in enumerate(zip(seq_rs, bwt_per_model)):
                offset = (i - len(seq_rs) / 2 + 0.5) * width6
                ax.bar(xt_bwt[:len(bwts)] + offset, bwts, width6,
                       label=r["model"], color=COLORS[i % len(COLORS)],
                       edgecolor="white", linewidth=0.4)
            ax.set_xticks(xt_bwt)
            ax.set_xticklabels(task_names[:n_tasks - 1], fontsize=8)
            ax.set_ylabel("BWT")
            ax.set_ylim(ymin_b, ymax_b)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.legend(fontsize=7, ncol=3, loc="upper right")
            plt.tight_layout()
            _fig_footer(fig)
            fig.savefig(os.path.join(plots_dir, f"per_task_bwt_{slug}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 7. Per-task FWT
        # FWT_j = R[j−1][j] − 0.5  for j = 1..T−1
        # how much prior training helps on a new task vs chance
        if seq_rs and n_tasks > 1:
            width7 = 0.8 / len(seq_rs)
            xt_fwt = np.arange(n_tasks - 1)
            all_fwt_vals = []
            fwt_per_model = []
            for r in seq_rs:
                R_raw = r["R_matrix"]
                T     = max(int(k) for k in R_raw) + 1
                fwts  = []
                for j in range(1, T):
                    rj1j = R_raw.get(str(j - 1), {}).get(str(j))
                    fwts.append((rj1j - 0.5) if rj1j is not None else 0.0)
                fwt_per_model.append(fwts)
                all_fwt_vals.extend(fwts)
            ymin_f = min(min(all_fwt_vals) - 0.05, -0.05)
            ymax_f = max(max(all_fwt_vals) + 0.05,  0.05)

            fig, ax = plt.subplots(
                figsize=(max(8, (n_tasks - 1) * len(seq_rs) * 0.35 + 2), 5))
            ax.set_title(
                f"Per-Task Forward Transfer (MH) — {ds}\n"
                "[FWT_j = R[j−1][j] − 0.5 | green = above chance, red = below chance]",
                fontweight="bold")
            ax.axhspan(0,      ymax_f, alpha=0.07, color="green", zorder=0)
            ax.axhspan(ymin_f, 0,      alpha=0.07, color="red",   zorder=0)
            for i, (r, fwts) in enumerate(zip(seq_rs, fwt_per_model)):
                offset = (i - len(seq_rs) / 2 + 0.5) * width7
                ax.bar(xt_fwt[:len(fwts)] + offset, fwts, width7,
                       label=r["model"], color=COLORS[i % len(COLORS)],
                       edgecolor="white", linewidth=0.4)
            ax.set_xticks(xt_fwt)
            ax.set_xticklabels(task_names[1:], fontsize=8)
            ax.set_ylabel("FWT")
            ax.set_ylim(ymin_f, ymax_f)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.legend(fontsize=7, ncol=3, loc="upper right")
            plt.tight_layout()
            _fig_footer(fig)
            fig.savefig(os.path.join(plots_dir, f"per_task_fwt_{slug}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 3. R-matrix heatmaps
    for r in all_results:
        R_raw = r.get("R_matrix", {})
        if not R_raw: continue
        n   = max(int(k) for k in R_raw) + 1
        mat = np.full((n, n), np.nan)
        for i_s, row in R_raw.items():
            for j_s, val in row.items():
                mat[int(i_s), int(j_s)] = val
        task_names = r["task_names"]
        fig, ax = plt.subplots(figsize=(max(4, n + 1), max(4, n + 1)))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(task_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(task_names, fontsize=8)
        ax.set_xlabel("Eval task")
        ax.set_ylabel("After task")
        ax.set_title(f"R-matrix: {r['model']} on {r['dataset']}", fontweight="bold")
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, color="black" if mat[i, j] > 0.4 else "white")
        slug_m = r["model"].lower().replace("-", "_")
        slug_d = r["dataset"].lower().replace("-", "").replace(" ", "")
        plt.tight_layout()
        _fig_footer(fig)
        fig.savefig(os.path.join(rmat_dir, f"{slug_m}_{slug_d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Per-config R-matrix heatmaps (one panel per task ordering)
    def _pair_lbl(p):
        return "/".join(str(x) for x in p) if isinstance(p, (list, tuple)) else str(p)
    for r in all_results:
        pc = r.get("per_config")
        if not pc:
            continue
        items   = list(pc.items())
        fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.4), squeeze=False)
        axes    = axes[0]
        last_im = None
        for ax, (cfg_name, cfg) in zip(axes, items):
            R_raw = cfg.get("R_matrix") or {}
            if not R_raw:
                ax.axis("off"); continue
            n   = max(int(k) for k in R_raw) + 1
            mat = np.full((n, n), np.nan)
            for i_s, row in R_raw.items():
                for j_s, val in row.items():
                    mat[int(i_s), int(j_s)] = val
            labels = [_pair_lbl(p) for p in cfg.get("task_pairs", list(range(n)))]
            if len(labels) != n:
                labels = [str(i + 1) for i in range(n)]
            last_im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Eval task"); ax.set_ylabel("After task")
            ax.set_title(f"{cfg_name}\n{' > '.join(labels)}", fontsize=9, fontweight="bold")
            for i in range(n):
                for j in range(n):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                                fontsize=7, color="black" if mat[i, j] > 0.4 else "white")
        fig.suptitle(f"R-matrix per config — {r['model']} on {r['dataset']}", fontweight="bold", y=1.03)
        if last_im is not None:
            fig.colorbar(last_im, ax=list(axes), fraction=0.025, pad=0.02)
        slug_m = r["model"].lower().replace("-", "_")
        slug_d = r["dataset"].lower().replace("-", "").replace(" ", "")
        _fig_footer(fig)
        fig.savefig(os.path.join(rmat_dir, f"{slug_m}_{slug_d}_per_config.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 9. MNIST pairing-config comparison plots
    multi_pc = [r for r in all_results if r.get("per_config")]
    mnist_pc = [r for r in multi_pc if r["dataset"] == "MNIST"]
    if mnist_pc:
        cfg_dir = os.path.join(plots_dir, "mnist_pairings")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_names  = list(MNIST_DATASET_CONFIGS.keys())
        cfg_colors = [COLORS[i % len(COLORS)] for i in range(len(cfg_names))]
        n_models   = len(mnist_pc)
        x          = np.arange(n_models)
        width      = 0.8 / len(cfg_names)
        model_labels = [r["model"] for r in mnist_pc]

        metric_specs = [
            ("final_accuracy",           "Final Accuracy"),
            ("final_backward_transfer",  "Final BWT"),
            ("average_backward_transfer","Avg BWT"),
            ("forward_transfer",         "Forward Transfer"),
            ("final_cohen_kappa",        "Cohen's Kappa"),
            ("plasticity",               "Plasticity"),
            ("stability",                "Stability"),
        ]
        for key, title in metric_specs:
            # Only models that define this metric (joint models report acc/kappa only)
            sub = [r for r in mnist_pc
                   if r["per_config"][cfg_names[0]]["summary"].get(key) is not None]
            if not sub:
                continue
            xs = np.arange(len(sub))
            sub_labels = [r["model"] for r in sub]
            fig, ax = plt.subplots(figsize=(max(8, len(sub) * len(cfg_names) * 0.45 + 2), 5))
            for ci, (cfg, col) in enumerate(zip(cfg_names, cfg_colors)):
                vals_raw = [r["per_config"][cfg]["summary"].get(key) for r in sub]
                vals = [v if v is not None else 0.0 for v in vals_raw]
                offset = (ci - len(cfg_names) / 2 + 0.5) * width
                bars = ax.bar(xs + offset, vals, width, label=f"{cfg}: " + ordering_label(MNIST_DATASET_CONFIGS[cfg], "mnist"),
                              color=col, edgecolor="white", linewidth=0.4)
                for bar, v in zip(bars, vals_raw):
                    if v is None:
                        bar.set_visible(False)
                        continue
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6)
            ax.set_xticks(xs)
            ax.set_xticklabels(sub_labels, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"MNIST {title} — per digit-pairing config", fontweight="bold")
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            ax.set_ylim(min(ax.get_ylim()[0], -0.05), max(ax.get_ylim()[1], 0.05))
            ax.legend(title="Pairing config", fontsize=8)
            plt.tight_layout()
            _fig_footer(fig)
            fig.savefig(os.path.join(cfg_dir, f"mnist_pairings_{key}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        metric_short = [
            ("final_accuracy",          "Acc_final"),
            ("final_backward_transfer", "BWT"),
            ("final_cohen_kappa",       "Kappa"),
            ("plasticity",              "Plasticity"),
            ("stability",               "Stability"),
        ]
        n_metrics = len(metric_short)
        for r in mnist_pc:
            fig, axes = plt.subplots(1, n_metrics,
                                     figsize=(n_metrics * 2.8, 4), sharey=False)
            fig.suptitle(f"MNIST pairing configs — {r['model']}", fontweight="bold")

            fig.text(0.5, -0.03, "   |   ".join(f"{c}: " + ordering_label(MNIST_DATASET_CONFIGS[c], "mnist") for c in cfg_names), ha="center", fontsize=7)
            for ax, (key, short) in zip(axes, metric_short):
                vals_raw = [r["per_config"][cfg]["summary"].get(key) for cfg in cfg_names]
                ax.set_title(short, fontsize=8)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                if all(v is None for v in vals_raw):     # metric undefined for this model
                    ax.set_xticks([])
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="#999999")
                    continue
                vals = [v if v is not None else 0.0 for v in vals_raw]
                bars = ax.bar(range(len(cfg_names)), vals,
                              color=cfg_colors, edgecolor="white", linewidth=0.5)
                ax.set_xticks(range(len(cfg_names)))
                ax.set_xticklabels(cfg_names, rotation=30, ha="right", fontsize=7)
                for bar, v in zip(bars, vals_raw):
                    if v is None:
                        bar.set_visible(False)
                        continue
                    va  = "bottom" if v >= 0 else "top"
                    off = 0.01    if v >= 0 else -0.01
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + off,
                            f"{v:.3f}", ha="center", va=va, fontsize=7)
            plt.tight_layout()
            slug_m = r["model"].lower().replace("-", "_")
            fig.savefig(os.path.join(cfg_dir, f"{slug_m}_pairings.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

# Main
TUNE_STUDIES = [
    # GIM (MNIST uses fixed paper params; WISDM is tuned)
    ("GIM-ALSTM on WISDM",   make_objective(lambda a,v,**kw: run_gim_wisdm_mh("alstm",a,v,**kw), to_args, gim_params, WISDM_DIR)),

    # LSTM baselines (naive + joint)
    ("LSTM on MNIST",        make_objective(run_naive_lstm_mnist_mh, to_args, lstm_params, MNIST_DIR)),
    ("LSTM on WISDM",        make_objective(run_naive_lstm_wisdm_mh, to_args, lstm_params, WISDM_DIR)),
    ("Joint-LSTM on MNIST",  make_objective(run_joint_lstm_mnist_mh, to_args, lstm_params, MNIST_DIR, pass_trial=False)),
    ("Joint-LSTM on WISDM",  make_objective(run_joint_lstm_wisdm_mh, to_args, lstm_params, WISDM_DIR, pass_trial=False)),
    # LSTM-Replay tuned independently per dataset
    ("LSTM-Replay on MNIST", make_objective(run_lstm_replay_mnist_mh, to_args, lstm_params, MNIST_DIR)),
    ("LSTM-Replay on WISDM", make_objective(run_lstm_replay_wisdm_mh, to_args, lstm_params, WISDM_DIR)),
    # ESN-Base: reservoir + training tuned once per dataset via Naive strategy
    ("ESN-Base on MNIST",    make_objective(run_esn_naive_mnist_mh, to_args, esn_base_params, MNIST_DIR)),
    ("ESN-Base on WISDM",    make_objective(run_esn_naive_wisdm_mh, to_args, esn_base_params, WISDM_DIR)),
    # Joint ESN: independently tuned (all tasks seen simultaneously)
    ("Joint-ESN on MNIST",   make_objective(run_joint_esn_mnist_mh, to_args, esn_base_params, MNIST_DIR, pass_trial=False)),
    ("Joint-ESN on WISDM",   make_objective(run_joint_esn_wisdm_mh, to_args, esn_base_params, WISDM_DIR, pass_trial=False)),

    # HHAR (TIL by device; per-device 6-way heads). Reuses the HAR (_wisdm) search spaces.
    ("GIM-ALSTM on HHAR",    make_objective(lambda a,v,**kw: run_gim_hhar_mh("alstm",a,v,**kw), to_args, gim_params, HHAR_DIR)),
    ("LSTM on HHAR",         make_objective(run_naive_lstm_hhar_mh, to_args, lstm_params, HHAR_DIR)),
    ("Joint-LSTM on HHAR",   make_objective(run_joint_lstm_hhar_mh, to_args, lstm_params, HHAR_DIR, pass_trial=False)),
    ("LSTM-Replay on HHAR",  make_objective(run_lstm_replay_hhar_mh, to_args, lstm_params, HHAR_DIR)),
    ("ESN-Base on HHAR",     make_objective(run_esn_naive_hhar_mh, to_args, esn_base_params, HHAR_DIR)),
    ("Joint-ESN on HHAR",    make_objective(run_joint_esn_hhar_mh, to_args, esn_base_params, HHAR_DIR, pass_trial=False)),
    # Fashion-MNIST studies (garment-vs-accessory DIL; GIM tuned like WISDM/HHAR)
    ("GIM-ALSTM on Fashion", make_objective(lambda a,v,**kw: run_gim_fashion_mh("alstm",a,v,**kw), to_args, gim_params, FASHION_DIR)),
    ("LSTM on Fashion",       make_objective(run_naive_lstm_fashion_mh, to_args, lstm_params, FASHION_DIR)),
    ("Joint-LSTM on Fashion", make_objective(run_joint_lstm_fashion_mh, to_args, lstm_params, FASHION_DIR, pass_trial=False)),
    ("LSTM-Replay on Fashion",make_objective(run_lstm_replay_fashion_mh, to_args, lstm_params, FASHION_DIR)),
    ("ESN-Base on Fashion",   make_objective(run_esn_naive_fashion_mh, to_args, esn_base_params, FASHION_DIR)),
    ("Joint-ESN on Fashion",  make_objective(run_joint_esn_fashion_mh, to_args, esn_base_params, FASHION_DIR, pass_trial=False)),
    # Fashion-MNIST Summer-vs-Winter (GIM uses fixed MNIST params, not tuned)
    ("GIM-ALSTM on Fashion-SW", make_objective(lambda a,v,**kw: run_gim_fashionsw_mh("alstm",a,v,**kw), to_args, gim_params, FASHION_DIR)),
    ("LSTM on Fashion-SW",       make_objective(run_naive_lstm_fashionsw_mh, to_args, lstm_params, FASHION_DIR)),
    ("Joint-LSTM on Fashion-SW", make_objective(run_joint_lstm_fashionsw_mh, to_args, lstm_params, FASHION_DIR, pass_trial=False)),
    ("LSTM-Replay on Fashion-SW",make_objective(run_lstm_replay_fashionsw_mh, to_args, lstm_params, FASHION_DIR)),
    ("ESN-Base on Fashion-SW",   make_objective(run_esn_naive_fashionsw_mh, to_args, esn_base_params, FASHION_DIR)),
    ("Joint-ESN on Fashion-SW",  make_objective(run_joint_esn_fashionsw_mh, to_args, esn_base_params, FASHION_DIR, pass_trial=False)),
]

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Multi-Head (TIL) CL pipeline")
    _ap.add_argument("--datasets", nargs="+", default=DATASETS,
                     choices=["mnist", "wisdm", "hhar", "fashion", "fashion_sw"],
                     help="Datasets to run (default: all). e.g. --datasets fashion")
    _cli = _ap.parse_args()
    DATASETS = _cli.datasets

    # Keep only the tuning studies for the selected datasets (study labels end
    # with ' on MNIST'/'WISDM'/'HHAR'), so skipped datasets never populate `tuned`.
    _sel_labels  = {DATASET_LABEL[d] for d in DATASETS}
    TUNE_STUDIES = [(l, o) for (l, o) in TUNE_STUDIES
                    if l.rsplit(" on ", 1)[1] in _sel_labels]

    _t_start = datetime.now()
    print(f"\n{'#'*70}\n#  MULTI-HEAD (TIL) EXPERIMENT\n{'#'*70}", flush=True)
    print(f"  started   : {_t_start:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  device    : {DEVICE}", flush=True)
    print(f"  datasets  : {DATASETS}", flush=True)
    print(f"  models    : {len(MODELS)}  ->  {MODELS}", flush=True)
    print(f"  total Phase-2 combinations: {len(DATASETS) * len(MODELS)}", flush=True)

    # Phase 1: Hyperparameter tuning
    primary_workers = min(len(TUNE_STUDIES), 8)
    print(f"\n{'='*70}\n  PHASE 1 — tuning {len(TUNE_STUDIES)} studies "
          f"(workers={primary_workers})\n{'='*70}", flush=True)
    _n = len(TUNE_STUDIES); _done = 0
    with ThreadPoolExecutor(max_workers=primary_workers) as ex:
        futs = {ex.submit(tune, label, obj): label for label, obj in TUNE_STUDIES}
        for fut in as_completed(futs):
            tuned[futs[fut]] = fut.result()
            _done += 1
            with PRINT_LOCK:
                print(f"  >> Phase 1 progress: {_done}/{_n} studies done", flush=True)

    # Built per dataset so only selected datasets reference `tuned` keys.
    model_dataset_params = {}
    if "mnist" in DATASETS:
        model_dataset_params.update({
            ("gim_alstm_mh",    "mnist"):  GIM_MNIST_PARAMS,
            ("lstm_mh",         "mnist"):  tuned["LSTM on MNIST"],
            ("joint_lstm_mh",   "mnist"):  tuned["Joint-LSTM on MNIST"],
            ("lstm_replay_mh",  "mnist"):  tuned["LSTM-Replay on MNIST"],
            ("esn_naive_mh",    "mnist"):  tuned["ESN-Base on MNIST"],
            ("joint_esn_mh",    "mnist"):  tuned["Joint-ESN on MNIST"],
        })
    if "wisdm" in DATASETS:
        model_dataset_params.update({
            ("gim_alstm_mh",    "wisdm"):  tuned["GIM-ALSTM on WISDM"],
            ("lstm_mh",         "wisdm"):  tuned["LSTM on WISDM"],
            ("joint_lstm_mh",   "wisdm"):  tuned["Joint-LSTM on WISDM"],
            ("lstm_replay_mh",  "wisdm"):  tuned["LSTM-Replay on WISDM"],
            ("esn_naive_mh",    "wisdm"):  tuned["ESN-Base on WISDM"],
            ("joint_esn_mh",    "wisdm"):  tuned["Joint-ESN on WISDM"],
        })
    if "hhar" in DATASETS:
        model_dataset_params.update({
            ("gim_alstm_mh",    "hhar"):   tuned["GIM-ALSTM on HHAR"],
            ("lstm_mh",         "hhar"):   tuned["LSTM on HHAR"],
            ("joint_lstm_mh",   "hhar"):   tuned["Joint-LSTM on HHAR"],
            ("lstm_replay_mh",  "hhar"):   tuned["LSTM-Replay on HHAR"],
            ("esn_naive_mh",    "hhar"):   tuned["ESN-Base on HHAR"],
            ("joint_esn_mh",    "hhar"):   tuned["Joint-ESN on HHAR"],
        })
    if "fashion" in DATASETS:
        model_dataset_params.update({
            # Fashion-MNIST (GIM fixed like MNIST)
            ("gim_alstm_mh",    "fashion"): tuned["GIM-ALSTM on Fashion"],
            ("lstm_mh",         "fashion"): tuned["LSTM on Fashion"],
            ("joint_lstm_mh",   "fashion"): tuned["Joint-LSTM on Fashion"],
            ("lstm_replay_mh",  "fashion"): tuned["LSTM-Replay on Fashion"],
            ("esn_naive_mh",    "fashion"): tuned["ESN-Base on Fashion"],
            ("joint_esn_mh",    "fashion"): tuned["Joint-ESN on Fashion"],
        })
    if "fashion_sw" in DATASETS:
        model_dataset_params.update({
            # Fashion-MNIST Summer-vs-Winter (GIM fixed like MNIST)
            ("gim_alstm_mh",    "fashion_sw"): tuned["GIM-ALSTM on Fashion-SW"],
            ("lstm_mh",         "fashion_sw"): tuned["LSTM on Fashion-SW"],
            ("joint_lstm_mh",   "fashion_sw"): tuned["Joint-LSTM on Fashion-SW"],
            ("lstm_replay_mh",  "fashion_sw"): tuned["LSTM-Replay on Fashion-SW"],
            ("esn_naive_mh",    "fashion_sw"): tuned["ESN-Base on Fashion-SW"],
            ("joint_esn_mh",    "fashion_sw"): tuned["Joint-ESN on Fashion-SW"],
        })

    # Phase 2: Full experiments
    exp_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"experiment_{exp_ts}")
    os.makedirs(exp_dir, exist_ok=True)

    combos = [(ds, m) for ds in DATASETS for m in MODELS]
    results_map = {}
    max_w = min(len(combos), os.cpu_count() or 4)
    print(f"\n{'='*70}\n  PHASE 2 — running {len(combos)} model/dataset combinations "
          f"(workers={max_w})\n{'='*70}", flush=True)
    _n = len(combos); _done = 0
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(run_combination, ds, m, model_dataset_params, exp_dir): (ds, m)
                for ds, m in combos}
        for fut in as_completed(futs):
            ds, m = futs[fut]
            results_map[(ds, m)] = fut.result()
            _done += 1
            with PRINT_LOCK:
                print(f"  >> Phase 2 progress: {_done}/{_n} combinations done "
                      f"(latest: {m} on {ds})", flush=True)
    all_results = [results_map[(ds, m)] for ds, m in combos]

    # Save results FIRST so hours of computation are never lost to a print/plot error
    save_results({
        "experiment":         exp_ts,
        "num_runs":           NUM_RUNS,
        "seed":               SEED,
        "subset_mnist":       SUBSET_MNIST,
        "subset_wisdm":       SUBSET_WISDM,
        "tune_subset_mnist":  TUNE_SUBSET_MNIST,
        "tune_subset_wisdm":  TUNE_SUBSET_WISDM,
        "results":            all_results,
    }, os.path.join(exp_dir, "summary.json"))

    save_plots(all_results, exp_dir)

    # Summary tables + completion banner
    print_metrics_table(all_results)
    print_hyperparams_table(all_results)
    _elapsed = datetime.now() - _t_start
    print(f"\n{'#'*70}\n#  DONE — multi-head experiment complete\n{'#'*70}", flush=True)
    print(f"  elapsed   : {_elapsed}", flush=True)
    print(f"  results   : {exp_dir}", flush=True)
