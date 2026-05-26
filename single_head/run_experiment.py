"""
Unified Continual Learning Pipeline.

Phase 1 -- Hyperparameter tuning (Optuna, TPE sampler):
  - GIM-ALSTM             : HAR  (MNIST uses original paper values)
  - LSTM / Joint-LSTM     : MNIST, HAR
  - ESN (each strat)      : MNIST, HAR  -- tuned independently per strategy
  - Joint-ESN             : MNIST, HAR

Phase 2 -- Full experiment with the best found parameters.
  Minimal output during runs; clean summary tables printed at the end.

Usage:
    python run_experiment.py
"""
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import threading
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
sys.path.insert(0, HERE)  # single_head/ must beat multi_head/ for 'models.*' lookups

from models.GIM_LSTM.gim_lstm_mnist_sh import run_gim_mnist
from models.GIM_LSTM.gim_lstm_wisdm_sh import run_gim_wisdm
from models.LSTM.naive.naive_lstm_mnist_sh import run_naive_lstm_mnist_sh
from models.LSTM.naive.naive_lstm_wisdm_sh import run_naive_lstm_wisdm_sh
from models.LSTM.joint.joint_lstm_mnist_sh import run_joint_lstm_mnist_sh
from models.LSTM.joint.joint_lstm_wisdm_sh import run_joint_lstm_wisdm_sh
from models.ESN.naive.naive_esn_mnist_sh import run_esn_naive_mnist
from models.ESN.naive.naive_esn_wisdm_sh import run_esn_naive_wisdm
from models.ESN.joint.joint_esn_mnist_sh import run_joint_esn_mnist_sh
from models.ESN.joint.joint_esn_wisdm_sh import run_joint_esn_wisdm_sh
from models.ESN.ewc.ewc_esn_mnist_sh import run_esn_ewc_mnist
from models.ESN.ewc.ewc_esn_wisdm_sh import run_esn_ewc_wisdm
from models.ESN.lwf.lwf_esn_mnist_sh import run_esn_lwf_mnist
from models.ESN.lwf.lwf_esn_wisdm_sh import run_esn_lwf_wisdm
from models.ESN.replay.replay_esn_mnist_sh import run_esn_replay_mnist
from models.ESN.replay.replay_esn_wisdm_sh import run_esn_replay_wisdm
from models.ESN.slda.slda_esn_mnist_sh import run_esn_slda_mnist
from models.ESN.slda.slda_esn_wisdm_sh import run_esn_slda_wisdm
from utils.metrics import save_results

# Avalanche's @deprecated decorator calls warnings.simplefilter("once") on every
# invocation, which prepends a filter that overrides our "ignore" filter.
# Unwrap the deprecated shell so the simplefilter call never fires.
try:
    from avalanche.training.storage_policy import ExemplarsBuffer
    if hasattr(ExemplarsBuffer.update, "__wrapped__"):
        ExemplarsBuffer.update = ExemplarsBuffer.update.__wrapped__
except Exception:
    pass

# ── Global settings ───────────────────────────────────────────────────
DATASETS    = ["mnist", "wisdm"]
MODELS      = ["gim_alstm_sh", "lstm_sh", "joint_lstm_sh",
                "esn_naive_sh", "esn_ewc_sh", "esn_lwf_sh", "esn_replay_sh",
                "esn_slda_sh", "joint_esn_sh"]
NUM_RUNS    = 1
SEED        = 42
RESULTS_DIR = os.path.join(HERE, "..", "results", "single_head")
MNIST_DIR   = os.path.join(HERE, "..", "data", "mnist")
WISDM_DIR     = os.path.join(HERE, "..", "data", "wisdm")
SUBSET_MNIST = 5000  # cap samples per task for MNIST; set None for full run
SUBSET_WISDM   = 2000  # cap samples per task for WISDM
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_LOCK = threading.Lock()

# ── Tuning settings ───────────────────────────────────────────────────
# N_TRIALS: TPE warmup is ~10 random trials; 15 gives 5 true exploitation steps on top.
# TUNE_SUBSET: GIM-HAR is the slowest study in Wave 1a (all run in parallel).
#   Phase 1a wall time ≈ N_TRIALS × (time per slowest trial).
N_TRIALS = 10  # trials for base / full studies
N_STRATEGY_TRIALS = 5  # trials for secondary ESN studies (1-2 params only, converges fast)
TUNE_SUBSET_MNIST = 500   # samples per task during MNIST tuning
TUNE_SUBSET_WISDM   = 200   # samples per task during WISDM tuning

# ── GIM on MNIST: original paper hyperparameters ─────────────────────
GIM_MNIST_PARAMS = Namespace(
    epochs                  = 7,       # original uses 2 on full data; 7 compensates for subset
    batch_size              = 32,      # original default
    learning_rate           = 3e-5,   # original paper default (mnist.py --learning_rate),     # original default,     # original default
    hidden_size_rnn         = 128,     # original default,     # original default
    hidden_sizes_lmn        = [128],   # original default
    hidden_size_autoencoder = 500,     # original paper default (mnist.py --hidden_size_autoencoder),      # downsampled for speed (original uses 28)
)

# ── Namespace builders from Optuna best_params ────────────────────────

def gim_namespace(best):
    return Namespace(
        hidden_sizes_lmn = [best["hidden_size_rnn"]],
        **best,
    )

def esn_namespace(best):
    defaults = dict(
        ewc_lambda      = 0.4,   # original smnist_ewc.yaml default,
        lwf_alpha       = 1.0,
        lwf_temperature = 1.0,   # original smnist_lwf.yaml default (was 2.0)
        mem_size        = 200,   # original smnist_replay.yaml default,  # StreamingLDA covariance regularisation
    )
    defaults.update(best)
    return Namespace(**defaults)

def lstm_namespace(best):
    return Namespace(**best)


# ── Optuna param samplers ─────────────────────────────────────────────

def gim_params_wisdm(trial):
    return dict(
        hidden_size_rnn         = trial.suggest_categorical("hidden_size_rnn", [32, 64, 128]),
        hidden_size_autoencoder = trial.suggest_categorical("hidden_size_autoencoder", [64, 128, 256]),
        learning_rate           = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size              = trial.suggest_categorical("batch_size", [16, 32, 64]),
        epochs                  = trial.suggest_int("epochs", 3, 7),
    )

def lstm_params_mnist(trial):
    return dict(
        hidden_size_rnn = trial.suggest_categorical("hidden_size_rnn", [64, 128, 256]),
        learning_rate   = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size      = trial.suggest_categorical("batch_size", [32, 64, 128]),
        epochs          = trial.suggest_int("epochs", 3, 7),
    )

def joint_lstm_params_mnist(trial):
    return dict(
        hidden_size_rnn = trial.suggest_categorical("hidden_size_rnn", [64, 128, 256]),
        learning_rate   = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size      = trial.suggest_categorical("batch_size", [32, 64, 128]),
        epochs          = trial.suggest_int("epochs", 2, 5),
    )

def lstm_params_wisdm(trial):
    return dict(
        hidden_size_rnn = trial.suggest_categorical("hidden_size_rnn", [32, 64, 128, 256]),
        learning_rate   = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        batch_size      = trial.suggest_categorical("batch_size", [16, 32, 64]),
        epochs          = trial.suggest_int("epochs", 3, 7),
    )

def esn_base_params_mnist(trial):
    return dict(
        esn_units     = trial.suggest_categorical("esn_units", [200, 500, 1000]),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128]),
        epochs        = trial.suggest_int("epochs", 2, 5),
    )

def esn_base_params_wisdm(trial):
    return dict(
        esn_units     = trial.suggest_categorical("esn_units", [100, 200, 500]),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64]),
        epochs        = trial.suggest_int("epochs", 3, 7),
    )

# Strategy-specific extra params only — reservoir + training params are shared from ESN-Base.
# Each secondary study tunes only 1-2 params on top of the fixed base.

def esn_ewc_extra(trial):
    return {"ewc_lambda": trial.suggest_categorical("ewc_lambda", [0.1, 1.0, 10.0])}

def esn_lwf_extra(trial):
    return {
        "lwf_alpha":       trial.suggest_float("lwf_alpha", 0.1, 5.0),
        "lwf_temperature": trial.suggest_categorical("lwf_temperature", [0.5, 1.0, 1.5]),
    }

def esn_replay_extra(trial):
    return {"mem_size": trial.suggest_categorical("mem_size", [50, 100])}


# ── Arg-builders for tuning ───────────────────────────────────────────

def gim_args(extra, **kwargs):
    return Namespace(
        device           = DEVICE,
        hidden_sizes_lmn = [extra["hidden_size_rnn"]],
        **extra, **kwargs,
    )

def esn_args(extra, **kwargs):
    defaults = dict(
        ewc_lambda      = 0.4,
        lwf_alpha       = 1.0,
        lwf_temperature = 1.0,
        mem_size        = 200,
        device          = DEVICE,
    )
    defaults.update(extra)
    defaults.update(kwargs)
    return Namespace(**defaults)

def lstm_args(extra, **kwargs):
    return Namespace(device=DEVICE, **extra, **kwargs)

# ── Optuna objectives ─────────────────────────────────────────────────
# `tuned` is populated after each Phase 1 study; secondary ESN objectives
# read it at call time — safe because base studies always run first.
tuned = {}

def make_objective(runner, arg_fn, param_fn, data_dir, base_key=None, pass_trial=True):
    """Return an Optuna objective that samples params, runs the experiment, and
    returns mean test accuracy.  base_key merges ESN-Base params for secondary studies."""
    tune_subset = TUNE_SUBSET_MNIST if data_dir == MNIST_DIR else TUNE_SUBSET_WISDM
    def objective(trial):
        params = param_fn(trial)
        if base_key is not None:
            params = {**tuned[base_key], **params}
        args = arg_fn(params, data_dir=data_dir, subset=tune_subset)
        trial_kwargs = {"trial": trial} if pass_trial else {}
        return float(np.mean(runner(args, False, **trial_kwargs)[0]))
    return objective

# ── Tuner ─────────────────────────────────────────────────────────────

def tune(label, objective, n_trials = None):
    n = n_trials if n_trials is not None else N_TRIALS
    with PRINT_LOCK:
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
    with PRINT_LOCK:
        print(f"  Done    {label:<40} best={study.best_value:.4f}")
    return study.best_params

# ── Task metadata & runners ───────────────────────────────────────────

TASK_META = {
    "mnist":   {"num_tasks": 5, "task_names": {0:"0/1",1:"2/3",2:"4/5",3:"6/7",4:"8/9"}},
    "wisdm":     {"num_tasks": 3, "task_names": {0:"Walk/Jog",1:"Up/Down",2:"Sit/Stand"}},
}

DATASET_LABEL = {"mnist": "MNIST", "wisdm": "WISDM"}

RUNNERS = {
    ("mnist",   "gim_alstm_sh"):    lambda a, v: run_gim_mnist("alstm", a, v),

    ("mnist",   "lstm_sh"):         lambda a, v: run_naive_lstm_mnist_sh(a, v),
    ("mnist",   "joint_lstm_sh"):   lambda a, v: run_joint_lstm_mnist_sh(a, v),
    ("mnist",   "esn_naive_sh"):    lambda a, v: run_esn_naive_mnist(a, v),
    ("mnist",   "esn_ewc_sh"):      lambda a, v: run_esn_ewc_mnist(a, v),
    ("mnist",   "esn_lwf_sh"):      lambda a, v: run_esn_lwf_mnist(a, v),
    ("mnist",   "esn_replay_sh"):   lambda a, v: run_esn_replay_mnist(a, v),
    ("mnist",   "esn_slda_sh"):     lambda a, v: run_esn_slda_mnist(a, v),
    ("mnist",   "joint_esn_sh"):    lambda a, v: run_joint_esn_mnist_sh(a, v),
    ("wisdm",     "gim_alstm_sh"):    lambda a, v: run_gim_wisdm("alstm", a, v),

    ("wisdm",     "lstm_sh"):         lambda a, v: run_naive_lstm_wisdm_sh(a, v),
    ("wisdm",     "joint_lstm_sh"):   lambda a, v: run_joint_lstm_wisdm_sh(a, v),
    ("wisdm",     "esn_naive_sh"):    lambda a, v: run_esn_naive_wisdm(a, v),
    ("wisdm",     "esn_ewc_sh"):      lambda a, v: run_esn_ewc_wisdm(a, v),
    ("wisdm",     "esn_lwf_sh"):      lambda a, v: run_esn_lwf_wisdm(a, v),
    ("wisdm",     "esn_replay_sh"):   lambda a, v: run_esn_replay_wisdm(a, v),
    ("wisdm",     "esn_slda_sh"):     lambda a, v: run_esn_slda_wisdm(a, v),
    ("wisdm",     "joint_esn_sh"):    lambda a, v: run_joint_esn_wisdm_sh(a, v),
}

SKIP_KEYS = {"num_runs", "seed", "results_dir", "subset", "data_dir", "device"}

def _build_args(model, dataset, model_dataset_params):
    args = Namespace(**vars(model_dataset_params[(model, dataset)]))
    args.num_runs    = NUM_RUNS
    args.seed        = SEED
    args.results_dir = RESULTS_DIR
    if dataset == "mnist":
        args.data_dir = MNIST_DIR
        args.subset   = SUBSET_MNIST
    elif dataset == "wisdm":
        args.data_dir = WISDM_DIR
        args.subset   = SUBSET_WISDM
    args.device = DEVICE
    return args

# ── Experiment runner (silent; returns result dict) ───────────────────

def run_combination(dataset, model, model_dataset_params,
                    exp_dir):
    meta          = TASK_META[dataset]
    num_tasks     = meta["num_tasks"]
    names         = meta["task_names"]
    runner        = RUNNERS[(dataset, model)]
    args          = _build_args(model, dataset, model_dataset_params)
    model_label   = model.upper().replace("_", "-")
    dataset_label = DATASET_LABEL[dataset]

    with PRINT_LOCK:
        print(f"  Running {model_label:<15} on {dataset_label:<12} ...", flush=True)

    all_val, all_test = [], []
    last_metrics = None
    for r in range(NUM_RUNS):
        torch.manual_seed(SEED + r)
        np.random.seed(SEED + r)
        result       = runner(args, False)
        all_val.append(result[0])
        all_test.append(result[1])
        last_metrics = result[2]

    all_val  = np.array(all_val)
    all_test = np.array(all_test)
    summary = last_metrics.summary()
    with PRINT_LOCK:
        print(f"  Done    {model_label:<15} on {dataset_label:<12}")

    hyperparams = {k: v for k, v in vars(args).items() if k not in SKIP_KEYS}
    out = {
        "model":       f"{model_label}-{dataset_label}",
        "dataset":     dataset,
        "model_id":    model,
        "num_runs":    NUM_RUNS,
        "hyperparams": hyperparams,
        "summary":     summary,
        "table": {
            names.get(t, str(t+1)): {
                "val_mean":  float(all_val.mean(0)[t]),
                "val_std":   float(all_val.std(0)[t]),
                "test_mean": float(all_test.mean(0)[t]),
                "test_std":  float(all_test.std(0)[t]),
            } for t in range(num_tasks)
        },
        "R_matrix": {
            str(i): {str(j): v for j, v in d.items()}
            for i, d in last_metrics.R.items()
        },
    }
    save_results(out, os.path.join(exp_dir, f"{model}_{dataset}.json"))

    return {
        "model":       model_label,
        "dataset":     dataset_label,
        "hyperparams": hyperparams,
        "summary":     summary,
        "val_mean":    all_val.mean(0).tolist(),
        "test_mean":   all_test.mean(0).tolist(),
        "task_names":  [names.get(t, str(t+1)) for t in range(num_tasks)],
        "R_matrix":    {str(i): {str(j): v for j, v in d.items()}
                        for i, d in last_metrics.R.items()},
    }

# ── Summary printers ──────────────────────────────────────────────────

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

# ── Plots ─────────────────────────────────────────────────────────────

def save_plots(all_results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

    plots_dir   = os.path.join(results_dir, "plots")
    rmat_dir    = os.path.join(plots_dir, "rmatrix")
    os.makedirs(rmat_dir, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in all_results))

    for ds in datasets:
        rs     = [r for r in all_results if r["dataset"] == ds]
        labels = [r["model"] for r in rs]
        x      = np.arange(len(rs))
        slug   = ds.lower().replace("-", "").replace(" ", "")

        # ── 1. CL metrics comparison ──────────────────────────────────
        metric_keys   = ["final_accuracy", "average_accuracy_over_time", "final_backward_transfer", "average_backward_transfer",
                         "forward_transfer", "final_cohen_kappa", "plasticity", "stability"]
        metric_titles = ["Acc Final", "Acc Avg", "BWT Final", "BWT Avg",
                         "FWT", "Cohen's kappa", "Plasticity", "Stability"]

        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        fig.suptitle(f"CL Metrics — {ds}", fontsize=12, fontweight="bold")

        for ax, key, title in zip(axes.flat, metric_keys, metric_titles):
            vals   = [r["summary"].get(key, 0.0) for r in rs]
            colors = [COLORS[i % len(COLORS)] for i in range(len(rs))]
            bars   = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
            ax.set_title(title)
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            ax.set_ylim(min(min(vals) - 0.05, -0.05), max(max(vals) + 0.05, 0.05))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"metrics_{slug}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── 2. Per-task final test accuracy ───────────────────────────
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
        ax.set_title(f"Per-Task Final Test Accuracy — {ds}", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", label="chance")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"per_task_{slug}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 3. R-matrix heatmaps ──────────────────────────────────────────
    for r in all_results:
        R_raw = r.get("R_matrix", {})
        if not R_raw:
            continue
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
        fig.savefig(os.path.join(rmat_dir, f"{slug_m}_{slug_d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")

# ── Main ──────────────────────────────────────────────────────────────

gim_runner = lambda args, verbose, **kwargs: run_gim_wisdm("alstm", args, verbose, **kwargs)

TUNE_STUDIES = [
    # GIM (MNIST uses fixed paper params; HAR is tuned)
    ("GIM-ALSTM on WISDM",    make_objective(gim_runner,           gim_args,  gim_params_wisdm,      WISDM_DIR)),

    # LSTM baselines (naive + joint, dataset-specific search spaces)
    ("LSTM on MNIST",       make_objective(run_naive_lstm_mnist_sh, lstm_args, lstm_params_mnist, MNIST_DIR)),
    ("LSTM on WISDM",         make_objective(run_naive_lstm_wisdm_sh,   lstm_args, lstm_params_wisdm,   WISDM_DIR)),
    ("Joint-LSTM on MNIST", make_objective(run_joint_lstm_mnist_sh, lstm_args, joint_lstm_params_mnist, MNIST_DIR, pass_trial=False)),
    ("Joint-LSTM on WISDM",   make_objective(run_joint_lstm_wisdm_sh,   lstm_args, lstm_params_wisdm,   WISDM_DIR,   pass_trial=False)),

    # ESN-Base: reservoir + training tuned once per dataset via Naive strategy
    ("ESN-Base on MNIST",   make_objective(run_esn_naive_mnist,     esn_args,  esn_base_params_mnist, MNIST_DIR)),
    ("ESN-Base on WISDM",     make_objective(run_esn_naive_wisdm,       esn_args,  esn_base_params_wisdm,   WISDM_DIR)),

    # ESN strategy-specific extras (N_STRATEGY_TRIALS each; must follow Base entries)
    ("ESN-EWC on MNIST",    make_objective(run_esn_ewc_mnist,   esn_args, esn_ewc_extra, MNIST_DIR, base_key="ESN-Base on MNIST")),
    ("ESN-EWC on WISDM",      make_objective(run_esn_ewc_wisdm,     esn_args, esn_ewc_extra, WISDM_DIR,   base_key="ESN-Base on WISDM")),
    ("ESN-LwF on MNIST",    make_objective(run_esn_lwf_mnist,    esn_args, esn_lwf_extra,    MNIST_DIR, base_key="ESN-Base on MNIST")),
    ("ESN-LwF on WISDM",      make_objective(run_esn_lwf_wisdm,      esn_args, esn_lwf_extra,    WISDM_DIR,   base_key="ESN-Base on WISDM")),
    ("ESN-Replay on MNIST", make_objective(run_esn_replay_mnist, esn_args, esn_replay_extra, MNIST_DIR, base_key="ESN-Base on MNIST")),
    ("ESN-Replay on WISDM",   make_objective(run_esn_replay_wisdm,   esn_args, esn_replay_extra, WISDM_DIR,   base_key="ESN-Base on WISDM")),

    # Joint ESN: independently tuned (all tasks seen simultaneously)
    ("Joint-ESN on MNIST",  make_objective(run_joint_esn_mnist_sh, esn_args, esn_base_params_mnist, MNIST_DIR, pass_trial=False)),
    ("Joint-ESN on WISDM",    make_objective(run_joint_esn_wisdm_sh,   esn_args, esn_base_params_wisdm,   WISDM_DIR,   pass_trial=False)),
]

if __name__ == "__main__":
    # ── Phase 1: Hyperparameter tuning ───────────────────────────────
    print(f"\n{'#'*70}")
    print("  Phase 1 -- Hyperparameter Tuning  "
          f"({N_TRIALS} trial(s), mnist_subset={TUNE_SUBSET_MNIST}, wisdm_subset={TUNE_SUBSET_WISDM})")
    print(f"{'#'*70}")

    # Phase 1a: independent studies in parallel (GIM, LSTM, Joint-LSTM, ESN-Base, Joint-ESN)
    # Phase 1b: ESN secondary studies (EWC/LwF) — must follow their ESN-Base
    SECONDARY = ("ESN-EWC", "ESN-LwF", "ESN-Replay")
    primary_studies   = [(l, o) for l, o in TUNE_STUDIES if not l.startswith(SECONDARY)]
    secondary_studies = [(l, o) for l, o in TUNE_STUDIES if l.startswith(SECONDARY)]

    primary_workers = min(len(primary_studies), 4)
    print(f"  [Phase 1a] {len(primary_studies)} independent studies, {primary_workers} workers")
    with ThreadPoolExecutor(max_workers=primary_workers) as ex:
        futs = {ex.submit(tune, label, obj): label for label, obj in primary_studies}
        for fut in as_completed(futs):
            tuned[futs[fut]] = fut.result()

    secondary_workers = min(len(secondary_studies), 8)
    print(f"  [Phase 1b] {len(secondary_studies)} secondary ESN studies, {secondary_workers} workers")
    with ThreadPoolExecutor(max_workers=secondary_workers) as ex:
        futs = {ex.submit(tune, label, obj, N_STRATEGY_TRIALS): label
                for label, obj in secondary_studies}
        for fut in as_completed(futs):
            label = futs[fut]
            ds = label.split(" on ", 1)[1]          # "MNIST", "HAR"
            tuned[label] = {**tuned[f"ESN-Base on {ds}"], **fut.result()}

    # ── Build per-(model, dataset) param Namespaces ───────────────────
    model_dataset_params = {
        # GIM
        ("gim_alstm_sh",    "mnist"):   GIM_MNIST_PARAMS,
        ("gim_alstm_sh",    "wisdm"):     gim_namespace(tuned["GIM-ALSTM on WISDM"]),

        # LSTM baselines
        ("lstm_sh",         "mnist"):   lstm_namespace(tuned["LSTM on MNIST"]),
        ("lstm_sh",         "wisdm"):     lstm_namespace(tuned["LSTM on WISDM"]),
        ("joint_lstm_sh",   "mnist"):   lstm_namespace(tuned["Joint-LSTM on MNIST"]),
        ("joint_lstm_sh",   "wisdm"):     lstm_namespace(tuned["Joint-LSTM on WISDM"]),
        # ESN: naive and slda share base reservoir params; ewc/lwf/replay carry merged dicts
        ("esn_naive_sh",    "mnist"):   esn_namespace(tuned["ESN-Base on MNIST"]),
        ("esn_naive_sh",    "wisdm"):     esn_namespace(tuned["ESN-Base on WISDM"]),
        ("esn_ewc_sh",      "mnist"):   esn_namespace(tuned["ESN-EWC on MNIST"]),
        ("esn_ewc_sh",      "wisdm"):     esn_namespace(tuned["ESN-EWC on WISDM"]),
        ("esn_lwf_sh",      "mnist"):   esn_namespace(tuned["ESN-LwF on MNIST"]),
        ("esn_lwf_sh",      "wisdm"):     esn_namespace(tuned["ESN-LwF on WISDM"]),
        ("esn_replay_sh",   "mnist"):   esn_namespace(tuned["ESN-Replay on MNIST"]),
        ("esn_replay_sh",   "wisdm"):     esn_namespace(tuned["ESN-Replay on WISDM"]),
        ("esn_slda_sh",     "mnist"):   esn_namespace(tuned["ESN-Base on MNIST"]),
        ("esn_slda_sh",     "wisdm"):     esn_namespace(tuned["ESN-Base on WISDM"]),
        ("joint_esn_sh",    "mnist"):   esn_namespace(tuned["Joint-ESN on MNIST"]),
        ("joint_esn_sh",    "wisdm"):     esn_namespace(tuned["Joint-ESN on WISDM"]),
    }

    # ── Phase 2: Full experiments ─────────────────────────────────────
    print(f"\n{'#'*70}")
    print("  Phase 2 -- Full Experiment  "
          f"({NUM_RUNS} run(s), seed={SEED})")
    print(f"{'#'*70}")

    exp_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"experiment_{exp_ts}")
    os.makedirs(exp_dir, exist_ok=True)

    combos = [(ds, m) for ds in DATASETS for m in MODELS]
    results_map = {}
    max_w = min(len(combos), os.cpu_count() or 4)
    print(f"  {len(combos)} combinations, {max_w} workers")
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(run_combination, ds, m, model_dataset_params, exp_dir): (ds, m)
                for ds, m in combos}
        for fut in as_completed(futs):
            ds, m = futs[fut]
            results_map[(ds, m)] = fut.result()
    all_results = [results_map[(ds, m)] for ds, m in combos]

    # ── Final summary tables, plots and summary.json ─────────────────
    print_hyperparams_table(all_results)
    print_metrics_table(all_results)
    save_plots(all_results, exp_dir)
    save_results({
        "experiment":    exp_ts,
        "num_runs":      NUM_RUNS,
        "seed":          SEED,
        "subset_mnist":  SUBSET_MNIST,
        "subset_wisdm":    SUBSET_WISDM,
        "results":       all_results,
    }, os.path.join(exp_dir, "summary.json"))

    print(f"\n{'#'*70}")
    print(f"  Done.  Results saved to {exp_dir}/")
    print(f"{'#'*70}\n")
