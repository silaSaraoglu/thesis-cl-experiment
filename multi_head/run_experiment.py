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
import os

os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
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
sys.path.insert(0, HERE)  # multi_head/ must beat single_head/ for 'models.*' lookups

from models.GIM_LSTM.gim_lstm_mnist_mh import run_gim_mnist_mh
from models.GIM_LSTM.gim_lstm_wisdm_mh import run_gim_wisdm_mh
from models.LSTM.naive.naive_lstm_mnist_mh import run_naive_lstm_mnist_mh
from models.LSTM.naive.naive_lstm_wisdm_mh import run_naive_lstm_wisdm_mh
from models.LSTM.joint.joint_lstm_mnist_mh import run_joint_lstm_mnist_mh
from models.LSTM.joint.joint_lstm_wisdm_mh import run_joint_lstm_wisdm_mh
from models.ESN.naive.naive_esn_mnist_mh import run_esn_naive_mnist_mh
from models.ESN.naive.naive_esn_wisdm_mh import run_esn_naive_wisdm_mh
from models.ESN.joint.joint_esn_mnist_mh import run_joint_esn_mnist_mh
from models.ESN.joint.joint_esn_wisdm_mh import run_joint_esn_wisdm_mh
from utils.metrics import save_results

# ── Global settings ───────────────────────────────────────────────────
DATASETS    = ["mnist", "wisdm"]
MODELS      = ["gim_alstm_mh", "lstm_mh", "joint_lstm_mh",
                "esn_naive_mh", "joint_esn_mh"]
NUM_RUNS    = 1
SEED        = 42
RESULTS_DIR = os.path.join(HERE, "..", "results", "multi_head")
MNIST_DIR   = os.path.join(HERE, "..", "data", "mnist")
WISDM_DIR     = os.path.join(HERE, "..", "data", "wisdm")
SUBSET_MNIST      = 5000
SUBSET_WISDM        = 2000
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_LOCK        = threading.Lock()

N_TRIALS          = 10
N_STRATEGY_TRIALS = 5
TUNE_SUBSET_MNIST = 500
TUNE_SUBSET_WISDM   = 500

# ── GIM on MNIST: original paper hyperparameters ──────────────────────
GIM_MNIST_PARAMS = Namespace(
    epochs                  = 7,
    batch_size              = 32,
    learning_rate           = 3e-5,
    hidden_size_rnn         = 128,
    hidden_sizes_lmn        = [128],
    hidden_size_autoencoder = 500,
    max_grad_norm           = 5.0,
)

# ── Namespace builders from Optuna best_params ────────────────────────

def gim_namespace(best):
    return Namespace(
        hidden_sizes_lmn = [best["hidden_size_rnn"]],
        max_grad_norm    = 5.0,
        **best,
    )

def esn_namespace(best):
    return Namespace(**best)

def lstm_namespace(best):
    return Namespace(**best)

# ── Optuna param samplers ─────────────────────────────────────────────

def gim_params_wisdm(trial):
    return dict(
        hidden_size_rnn         = trial.suggest_categorical("hidden_size_rnn", [64, 128, 256, 512]),
        hidden_size_autoencoder = trial.suggest_categorical("hidden_size_autoencoder", [64, 128, 256, 500, 512, 1024]),
        learning_rate           = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size              = trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        epochs                  = trial.suggest_int("epochs", 3, 10),
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
        batch_size      = trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        epochs          = trial.suggest_int("epochs", 3, 10),
    )

def esn_base_params_mnist(trial):
    return dict(
        esn_units     = trial.suggest_categorical("esn_units", [200, 500, 1000]),
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128]),
        epochs        = trial.suggest_int("epochs", 2, 5),
    )

def esn_base_params_wisdm(trial):
    return dict(
        esn_units     = trial.suggest_categorical("esn_units", [100, 200, 500]),
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64]),
        epochs        = trial.suggest_int("epochs", 3, 10),
    )

# ── Arg-builders for tuning ───────────────────────────────────────────

def gim_args(extra, subset, **kwargs):
    return Namespace(
        subset           = subset,
        device           = DEVICE,
        hidden_sizes_lmn = [extra["hidden_size_rnn"]],
        max_grad_norm    = 5.0,
        **extra, **kwargs,
    )

def esn_args(extra, subset, **kwargs):
    defaults = dict(
        subset = subset,
        device = DEVICE,
    )
    defaults.update(extra)
    defaults.update(kwargs)
    return Namespace(**defaults)

def lstm_args(extra, subset, **kwargs):
    return Namespace(device=DEVICE, subset=subset, **extra, **kwargs)

# ── Optuna objectives ─────────────────────────────────────────────────
tuned = {}

def objective_gim_alstm_wisdm(trial):
    val_accs, _, _, _, _ = run_gim_wisdm_mh(
        "alstm", gim_args(gim_params_wisdm(trial), subset=TUNE_SUBSET_WISDM, data_dir=WISDM_DIR),
        verbose=False, trial=trial)
    return float(np.mean(val_accs))

def objective_lstm_mnist(trial):
    val_accs, _, _, _, _ = run_naive_lstm_mnist_mh(
        lstm_args(lstm_params_mnist(trial), subset=TUNE_SUBSET_MNIST, data_dir=MNIST_DIR),
        verbose=False, trial=trial)
    return float(np.mean(val_accs))

def objective_lstm_wisdm(trial):
    val_accs, _, _, _, _ = run_naive_lstm_wisdm_mh(
        lstm_args(lstm_params_wisdm(trial), subset=TUNE_SUBSET_WISDM, data_dir=WISDM_DIR),
        verbose=False, trial=trial)
    return float(np.mean(val_accs))

def objective_joint_lstm_mnist(trial):
    val_accs, _, _, _, _ = run_joint_lstm_mnist_mh(
        lstm_args(joint_lstm_params_mnist(trial), subset=TUNE_SUBSET_MNIST, data_dir=MNIST_DIR),
        verbose=False)
    return float(np.mean(val_accs))

def objective_joint_lstm_wisdm(trial):
    val_accs, _, _, _, _ = run_joint_lstm_wisdm_mh(
        lstm_args(lstm_params_wisdm(trial), subset=TUNE_SUBSET_WISDM, data_dir=WISDM_DIR),
        verbose=False)
    return float(np.mean(val_accs))

def objective_joint_esn_mnist(trial):
    val_accs, _, _ = run_joint_esn_mnist_mh(
        esn_args(esn_base_params_mnist(trial), subset=TUNE_SUBSET_MNIST, data_dir=MNIST_DIR),
        verbose=False)
    return float(np.mean(val_accs))

def objective_joint_esn_wisdm(trial):
    val_accs, _, _ = run_joint_esn_wisdm_mh(
        esn_args(esn_base_params_wisdm(trial), subset=TUNE_SUBSET_WISDM, data_dir=WISDM_DIR),
        verbose=False)
    return float(np.mean(val_accs))

def objective_esn_base_mnist(trial):
    val_accs, _, _ = run_esn_naive_mnist_mh(
        esn_args(esn_base_params_mnist(trial), subset=TUNE_SUBSET_MNIST, data_dir=MNIST_DIR),
        verbose=False, trial=trial)
    return float(np.mean(val_accs))

def objective_esn_base_wisdm(trial):
    val_accs, _, _ = run_esn_naive_wisdm_mh(
        esn_args(esn_base_params_wisdm(trial), subset=TUNE_SUBSET_WISDM, data_dir=WISDM_DIR),
        verbose=False, trial=trial)
    return float(np.mean(val_accs))

# ── Tuner ─────────────────────────────────────────────────────────────

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
        print(f"  Done    {label:<40} best={study.best_value:.4f}")
    return study.best_params

# ── Task metadata & runners ───────────────────────────────────────────

TASK_META = {
    "mnist":   {"num_tasks": 5, "task_names": {0:"0/1",1:"2/3",2:"4/5",3:"6/7",4:"8/9"}},
    "wisdm":     {"num_tasks": 3, "task_names": {0:"Walk/Jog",1:"Up/Down",2:"Sit/Stand"}},
}

DATASET_LABEL = {"mnist": "MNIST", "wisdm": "WISDM"}

RUNNERS = {
    ("mnist", "gim_alstm_mh"):    lambda a, v: run_gim_mnist_mh("alstm", a, v),

    ("mnist", "lstm_mh"):         lambda a, v: run_naive_lstm_mnist_mh(a, v),
    ("mnist", "joint_lstm_mh"):   lambda a, v: run_joint_lstm_mnist_mh(a, v),
    ("mnist", "esn_naive_mh"):    lambda a, v: run_esn_naive_mnist_mh(a, v),
    ("mnist", "joint_esn_mh"):    lambda a, v: run_joint_esn_mnist_mh(a, v),
    ("wisdm",   "gim_alstm_mh"):    lambda a, v: run_gim_wisdm_mh("alstm", a, v),

    ("wisdm",   "lstm_mh"):         lambda a, v: run_naive_lstm_wisdm_mh(a, v),
    ("wisdm",   "joint_lstm_mh"):   lambda a, v: run_joint_lstm_wisdm_mh(a, v),
    ("wisdm",   "esn_naive_mh"):    lambda a, v: run_esn_naive_wisdm_mh(a, v),
    ("wisdm",   "joint_esn_mh"):    lambda a, v: run_joint_esn_wisdm_mh(a, v),
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
    save_results({
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
    }, os.path.join(exp_dir, f"{model}_{dataset}.json"))

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
    print(f"\n{'='*70}\n  BEST HYPERPARAMETERS\n{'='*70}")
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

    print(f"\n{'='*110}\n  CL METRICS SUMMARY\n{'='*110}")
    header = "  " + "  ".join(f"{c:<{col_widths[i]}}" for i, c in enumerate(cols))
    print(header); print(sep)
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

    print(f"\n{'='*90}\n  PER-TASK TEST ACCURACY\n{'='*90}")
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
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

    plots_dir = os.path.join(results_dir, "plots")
    rmat_dir  = os.path.join(plots_dir, "rmatrix")
    os.makedirs(rmat_dir, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in all_results))

    for ds in datasets:
        rs     = [r for r in all_results if r["dataset"] == ds]
        labels = [r["model"] for r in rs]
        x      = np.arange(len(rs))
        slug   = ds.lower().replace("-", "").replace(" ", "")

        metric_keys   = ["final_accuracy", "average_accuracy_over_time", "final_backward_transfer", "average_backward_transfer",
                         "forward_transfer", "final_cohen_kappa", "plasticity", "stability"]
        metric_titles = ["Acc Final", "Acc Avg", "BWT Final", "BWT Avg",
                         "FWT", "Cohen's kappa", "Plasticity", "Stability"]

        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        fig.suptitle(f"CL Metrics (MH) — {ds}", fontsize=12, fontweight="bold")
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
        ax.set_xticks(xt); ax.set_xticklabels(task_names, fontsize=8)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Per-Task Final Test Accuracy (MH) — {ds}", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", label="chance")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"per_task_{slug}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

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
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(task_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(task_names, fontsize=8)
        ax.set_xlabel("Eval task"); ax.set_ylabel("After task")
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

TUNE_STUDIES = [
    # GIM (MNIST uses fixed paper params; HAR is tuned)
    ("GIM-ALSTM on WISDM",      objective_gim_alstm_wisdm),

    # LSTM baselines (naive + joint share the same search space)
    ("LSTM on MNIST",         objective_lstm_mnist),
    ("LSTM on WISDM",           objective_lstm_wisdm),
    ("Joint-LSTM on MNIST",   objective_joint_lstm_mnist),
    ("Joint-LSTM on WISDM",     objective_joint_lstm_wisdm),
    # ESN-Base: reservoir + training tuned once per dataset via Naive strategy
    ("ESN-Base on MNIST",     objective_esn_base_mnist),
    ("ESN-Base on WISDM",       objective_esn_base_wisdm),
    # Joint ESN: independently tuned (all tasks seen simultaneously)
    ("Joint-ESN on MNIST",    objective_joint_esn_mnist),
    ("Joint-ESN on WISDM",      objective_joint_esn_wisdm),
]

if __name__ == "__main__":
    print(f"\n{'#'*70}")
    print("  Phase 1 -- Hyperparameter Tuning  "
          f"({N_TRIALS} trial(s), "
          f"subset_mnist={TUNE_SUBSET_MNIST}, subset_wisdm={TUNE_SUBSET_WISDM})")
    print(f"{'#'*70}")

    primary_workers = min(len(TUNE_STUDIES), 4)
    print(f"  [Phase 1a] {len(TUNE_STUDIES)} independent studies, {primary_workers} workers")
    with ThreadPoolExecutor(max_workers=primary_workers) as ex:
        futs = {ex.submit(tune, label, obj): label for label, obj in TUNE_STUDIES}
        for fut in as_completed(futs):
            tuned[futs[fut]] = fut.result()

    model_dataset_params = {
        # GIM
        ("gim_alstm_mh",   "mnist"):  GIM_MNIST_PARAMS,
        ("gim_alstm_mh",   "wisdm"):    gim_namespace(tuned["GIM-ALSTM on WISDM"]),

        # LSTM baselines
        ("lstm_mh",        "mnist"):  lstm_namespace(tuned["LSTM on MNIST"]),
        ("lstm_mh",        "wisdm"):    lstm_namespace(tuned["LSTM on WISDM"]),
        ("joint_lstm_mh",  "mnist"):  lstm_namespace(tuned["Joint-LSTM on MNIST"]),
        ("joint_lstm_mh",  "wisdm"):    lstm_namespace(tuned["Joint-LSTM on WISDM"]),
        # ESN baselines
        ("esn_naive_mh",   "mnist"):  esn_namespace(tuned["ESN-Base on MNIST"]),
        ("esn_naive_mh",   "wisdm"):    esn_namespace(tuned["ESN-Base on WISDM"]),
        ("joint_esn_mh",   "mnist"):  esn_namespace(tuned["Joint-ESN on MNIST"]),
        ("joint_esn_mh",   "wisdm"):    esn_namespace(tuned["Joint-ESN on WISDM"]),
    }

    print(f"\n{'#'*70}")
    print(f"  Phase 2 -- Full Experiment  ({NUM_RUNS} run(s), seed={SEED})")
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
