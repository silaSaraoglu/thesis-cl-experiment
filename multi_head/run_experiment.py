"""
Multi-Head (TIL Setting 2) Continual Learning Pipeline.

Architecture: per-task linear heads via ModuleDict.  Oracle task ID provided
at test time — this is the Task-Incremental Learning (TIL) upper bound.

Phase 1 -- Hyperparameter tuning (same search spaces as SH), over MNIST, WISDM,
HHAR and Fashion-SW:
  - GIM-ALSTM                       : WISDM, HHAR, Fashion-SW (MNIST uses fixed paper params)
  - LSTM / Joint-LSTM / LSTM-Replay : all datasets
  - ESN-Base (Naive MH) / Joint-ESN : all datasets

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
import numpy as np
import torch
from argparse import Namespace
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
_p = HERE
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import shared.utils
from shared.utils import load_tuned_params
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
from models.GIM_LSTM.gim_lstm_fashionsw_mh import run_gim_fashionsw_mh
from models.LSTM.naive.naive_lstm_fashionsw_mh import run_naive_lstm_fashionsw_mh
from models.LSTM.joint.joint_lstm_fashionsw_mh import run_joint_lstm_fashionsw_mh
from models.LSTM.replay.replay_lstm_fashionsw_mh import run_lstm_replay_fashionsw_mh
from models.ESN.naive.naive_esn_fashionsw_mh import run_esn_naive_fashionsw_mh
from models.ESN.joint.joint_esn_fashionsw_mh import run_joint_esn_fashionsw_mh
from shared.metrics import save_results
from shared.plot_style import (
    FS_ANNOT, FS_LEGEND, FS_SUPTITLE, FS_TICK, FS_TITLE,
    apply_thesis_style, config_colors, fig_size, grid_size, label_bars,
    model_legend, save_fig, zero_bar_stubs,
)
from config import * 

TASK_META = {
    "mnist":      {"num_tasks": 5},
    "wisdm":      {"num_tasks": 3},
    "hhar":       {"num_tasks": 4, "task_names": {0:"nexus4",1:"s3",2:"s3mini",3:"samsungold"}},
    "fashion_sw": {"num_tasks": 3},
}

DATASET_LABEL = {"mnist": "MNIST", "wisdm": "WISDM", "hhar": "HHAR", "fashion_sw": "Fashion-SW"}

SUBSETS      = {"mnist": SUBSET_MNIST, "wisdm": SUBSET_WISDM,
                "hhar": SUBSET_HHAR, "fashion_sw": SUBSET_FASHION}
TUNE_SUBSETS = {"mnist": TUNE_SUBSET_MNIST, "wisdm": TUNE_SUBSET_WISDM,
                "hhar": TUNE_SUBSET_HHAR, "fashion_sw": TUNE_SUBSET_FASHION}

def subset_meta(datasets):
    """Run-metadata block describing the train/tune subset caps for ``datasets``."""
    return {
        "subset":      {ds: SUBSETS[ds]      for ds in datasets},
        "tune_subset": {ds: TUNE_SUBSETS[ds] for ds in datasets},
    }

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
    ("fashion_sw","gim_alstm_mh"):    lambda a, v: run_gim_fashionsw_mh("alstm", a, v),
    ("fashion_sw","lstm_mh"):         lambda a, v: run_naive_lstm_fashionsw_mh(a, v),
    ("fashion_sw","joint_lstm_mh"):   lambda a, v: run_joint_lstm_fashionsw_mh(a, v),
    ("fashion_sw","lstm_replay_mh"):  lambda a, v: run_lstm_replay_fashionsw_mh(a, v),
    ("fashion_sw","esn_naive_mh"):    lambda a, v: run_esn_naive_fashionsw_mh(a, v),
    ("fashion_sw","joint_esn_mh"):    lambda a, v: run_joint_esn_fashionsw_mh(a, v),
}

SKIP_KEYS = {"results_dir", "subset", "data_dir", "device",
             "holdout_n", "holdout_seed", "use_holdout"}

def build_args(model, dataset, model_dataset_params):
    args = Namespace(**model_dataset_params[(model, dataset)])
    args.results_dir = RESULTS_DIR
    if dataset == "mnist":
        args.data_dir    = MNIST_DIR
        args.subset      = SUBSET_MNIST
    elif dataset == "wisdm":
        args.data_dir    = WISDM_DIR
        args.subset      = SUBSET_WISDM
    elif dataset == "hhar":
        args.data_dir    = HHAR_DIR
        args.subset      = SUBSET_HHAR
    elif dataset == "fashion_sw":
        args.data_dir    = FASHION_DIR
        args.subset      = SUBSET_FASHION

    args.holdout_n    = 0
    args.holdout_seed = TUNE_SPLIT_SEED
    args.use_holdout  = False
    args.device       = DEVICE
    return args

def run_combination(dataset, model, model_dataset_params, exp_dir, epochs=None, config_names=None):
    runner        = RUNNERS[(dataset, model)]
    args          = build_args(model, dataset, model_dataset_params)
    if epochs is not None:
        args.epochs = epochs
    model_label   = model.upper().replace("_", "-")
    dataset_label = DATASET_LABEL[dataset]

    print(f"  Running {model_label:<15} on {dataset_label:<12} ...", flush=True)

    hyperparams = {k: v for k, v in vars(args).items() if k not in SKIP_KEYS}

    # Every dataset defines task configs; run once per config (results stay per-config, no averaging).
    task_cfgs = DATASET_CONFIGS[dataset]
    cfg_names = config_names if config_names is not None else list(task_cfgs)
    per_config = {}
    for cfg_name in cfg_names:
        task_pairs = task_cfgs[cfg_name]
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        run_args = Namespace(**vars(args), task_pairs=task_pairs)
        result   = runner(run_args, False)
        # Forward transfer is not meaningful in the multi-head (TIL) setting
        # — each task has its own untrained head, so it's dropped here.
        summ = result[2].summary()
        summ.pop("forward_transfer", None)
        summ.pop("forward_transfer_kappa", None)
        per_config[cfg_name] = {
            "val":      [float(v) for v in result[0]],
            "test":     [float(v) for v in result[1]],
            "summary":  summ,
            "R_matrix": {str(i): {str(j): v for j, v in d.items()}
                         for i, d in result[2].R.items()},
            # Kappa counterpart of R_matrix: every model records a Kappa per
            # (after_task, eval_task), so persisting it keeps the whole Kappa metric
            # suite recomputable from the JSONs without re-training.
            "K_matrix": {str(i): {str(j): v for j, v in d.items()}
                         for i, d in result[2].K.items()},
            "task_pairs": task_pairs,
        }

    out = {
        "model":       f"{model_label}-{dataset_label}",
        "dataset":     dataset,
        "model_id":    model,
        "hyperparams": hyperparams,
        "per_config":  per_config,
    }
    save_results(out, os.path.join(exp_dir, f"{model}_{dataset}.json"))

    return {
        "model":       model_label,
        "dataset":     dataset_label,
        "hyperparams": hyperparams,
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


# ---- plotting / printing helpers --------------------------------------------------

def pair_lbl(p):
    return "/".join(str(x) for x in p) if isinstance(p, (list, tuple)) else str(p)

def dense(d):
    if not d:
        return None, 0
    n = max(int(k) for k in d) + 1
    mat = np.full((n, n), np.nan)
    for i_s, row in d.items():
        for j_s, v in row.items():
            mat[int(i_s), int(j_s)] = v
    return mat, n

def cfg_summary(r, cfg):
    return r["per_config"][cfg]["summary"]

def cfg_R(r, cfg):
    return r["per_config"][cfg]["R_matrix"]

def cfg_K(r, cfg):
    """Task x task Kappa matrix; {} if it was not recorded."""
    return r["per_config"][cfg].get("K_matrix", {})

def cfg_test(r, cfg):
    return r["per_config"][cfg]["test"]

def agg_metric(r, key):
    """Mean and std of ``key`` across a result's task-configs; (None, None) if absent.
    Mirrors shared.epoch_analysis.mean_std so the main experiment and the epoch sweep
    aggregate configs the same way."""
    pc = r.get("per_config") or {}
    vals = [c["summary"].get(key) for c in pc.values()]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))

def fig_footer(fig, present):
    train_bits, tune_bits = [], []
    for ds in present:
        s, t = SUBSETS[ds], TUNE_SUBSETS[ds]
        train_bits.append(f"{DATASET_LABEL[ds]}={s if s is not None else 'full'}")
        tune_bits.append(f"{DATASET_LABEL[ds]}={t if t is not None else 'full'}")
    parts = []
    if train_bits:
        parts.append("Train subset: " + ", ".join(train_bits))
    if tune_bits:
        parts.append("Tune subset: " + ", ".join(tune_bits))
    if parts:
        fig.text(0.5, 0.002, "  |  ".join(parts), ha="center", va="bottom",
                 fontsize=FS_FOOTER, color="#777777", style="italic")


def print_metrics_table(all_results):
    cols = ["Model", "Dataset", "Acc_final", "Acc_avg", "BWT_final", "BWT_avg",
            "Kappa", "Plasticity", "Stability"]
    col_widths = [15, 12, 10, 10, 10, 10, 8, 11, 10]
    sep  = "  " + "  ".join("-" * c for c in col_widths)

    # ordered union of config names across results
    cfg_names = []
    for r in all_results:
        for c in (r.get("per_config") or {}):
            if c not in cfg_names:
                cfg_names.append(c)

    for cfg in cfg_names:
        print(f"\n{'='*110}")
        print(f"  CL METRICS SUMMARY  --  {cfg}")
        print(f"{'='*110}")
        print("  " + "  ".join(f"{c:<{col_widths[i]}}" for i, c in enumerate(cols)))
        print(sep)
        for r in all_results:
            pc = r.get("per_config") or {}
            if cfg not in pc:
                continue
            s = pc[cfg]["summary"]
            row = [
                r["model"], r["dataset"],
                format_metric(s.get('final_accuracy')),            format_metric(s.get('average_accuracy_over_time')),
                format_metric(s.get('final_backward_transfer')),   format_metric(s.get('average_backward_transfer')),
                format_metric(s.get('final_cohen_kappa')),
                format_metric(s.get('plasticity')),                format_metric(s.get('stability')),
            ]
            print("  " + "  ".join(f"{v:<{col_widths[i]}}" for i, v in enumerate(row)))
        print(sep)

    for cfg in cfg_names:
        print(f"\n{'='*90}")
        print(f"  PER-TASK TEST ACCURACY  --  {cfg}")
        print(f"{'='*90}")
        for r in all_results:
            pc = r.get("per_config") or {}
            if cfg not in pc:
                continue
            labels = [pair_lbl(p) for p in pc[cfg]["task_pairs"]]
            task_str = "  ".join(f"{n}: {a:.4f}" for n, a in zip(labels, pc[cfg]["test"]))
            print(f"  {r['model']:<15} {r['dataset']:<12}  {task_str}")

    # Aggregate over configs -- mean +/- std per (model, dataset) across all configs.
    metric_order = ["final_accuracy", "average_accuracy_over_time",
                    "final_backward_transfer", "average_backward_transfer",
                    "final_cohen_kappa", "plasticity", "stability"]
    agg_widths = [15, 12] + [13] * len(metric_order)
    agg_sep = "  " + "  ".join("-" * c for c in agg_widths)
    datasets = []
    for r in all_results:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    for ds in datasets:
        ds_rs = [r for r in all_results if r["dataset"] == ds]
        n_cfg = max((len(r.get("per_config") or {}) for r in ds_rs), default=0)
        if n_cfg < 2:        # nothing to aggregate with a single config
            continue
        print(f"\n{'='*120}")
        print(f"  CL METRICS  --  {ds}  (mean +/- std over {n_cfg} configs)")
        print(f"{'='*120}")
        print("  " + "  ".join(f"{c:<{agg_widths[i]}}" for i, c in enumerate(cols)))
        print(agg_sep)
        for r in ds_rs:
            cells = [r["model"], r["dataset"]]
            for key in metric_order:
                mu, sd = agg_metric(r, key)
                cells.append("     N/A     " if mu is None else f"{mu:.3f}±{sd:.3f}")
            print("  " + "  ".join(f"{v:<{agg_widths[i]}}" for i, v in enumerate(cells)))
        print(agg_sep)

def save_plots(all_results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_thesis_style()
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Datasets present in these results, in first-seen order (footer lists each one's caps).
    _label_to_ds = {v: k for k, v in DATASET_LABEL.items()}
    _present = []
    for r in all_results:
        ds = _label_to_ds.get(r["dataset"])
        if ds and ds not in _present:
            _present.append(ds)

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metric_keys = [
        "final_accuracy", "average_accuracy_over_time",
        "final_backward_transfer", "average_backward_transfer",
        "final_cohen_kappa", "plasticity", "stability",
    ]
    metric_titles = [
        ("Final Accuracy",         "mean R[T-1][j] over all j"),
        ("Avg Accuracy Over Time", "mean R[i][j] for all i>=j"),
        ("Final BWT",              "mean(R[T-1][j]-R[j][j]) for j<T-1"),
        ("Avg BWT",                "mean(R[i][j]-R[j][j]) for all i>j"),
        ("Cohen's Kappa",          "agreement above chance"),
        ("Plasticity",             "mean diagonal R[j][j]"),
        ("Stability",              "mean off-diag (i>j)"),
    ]

    # Kappa-valued counterparts, computed from the K-matrix exactly as the accuracy
    # metrics are from the R-matrix. Same panel order as above so the accuracy figure
    # and the Kappa figure can be read side by side.
    kappa_metric_keys = [
        "final_cohen_kappa", "average_kappa_over_time",
        "final_backward_transfer_kappa", "average_backward_transfer_kappa",
        "plasticity_kappa", "stability_kappa",
    ]
    kappa_metric_titles = [
        ("Final Kappa",           "mean K[T-1][j] over all j"),
        ("Avg Kappa Over Time",   "mean K[i][j] for all i>=j"),
        ("Final BWT (Kappa)",     "mean(K[T-1][j]-K[j][j]) for j<T-1"),
        ("Avg BWT (Kappa)",       "mean(K[i][j]-K[j][j]) for all i>j"),
        ("Plasticity (Kappa)",    "mean diagonal K[j][j]"),
        ("Stability (Kappa)",     "mean off-diag (i>j)"),
    ]

    # (keys, titles, filename, figure title, probe key) -- one figure per set; a set is
    # skipped when its probe key is missing from the results.
    metric_sets = [
        (metric_keys,       metric_titles,       "metrics.png",
         "CL Metrics",                 "final_accuracy"),
        (kappa_metric_keys, kappa_metric_titles, "metrics_kappa.png",
         "CL Metrics (Cohen's Kappa)", "average_kappa_over_time"),
    ]

    for ds in sorted(set(r["dataset"] for r in all_results)):
        ds_rs   = [r for r in all_results if r["dataset"] == ds]
        ds_slug = ds.lower().replace("-", "_").replace(" ", "")
        chance  = 1.0 / 6 if "hhar" in ds.lower() else 0.5   # per-task chance line = 1/num_classes
        cfgs    = list(ds_rs[0]["per_config"].keys())

        for cfg in cfgs:
            cdir  = os.path.join(plots_dir, ds_slug, cfg)
            rdir  = os.path.join(cdir, "rmatrix")
            kdir  = os.path.join(cdir, "kmatrix")
            os.makedirs(rdir, exist_ok=True)
            os.makedirs(kdir, exist_ok=True)

            task_names = [pair_lbl(p) for p in ds_rs[0]["per_config"][cfg].get("task_pairs", [])]
            n_tasks = len(task_names)
            labels  = [r["model"] for r in ds_rs]
            tag     = f"  [{cfg}]"

            # 1) CL-metrics overview (only models that define each metric).
            #    Emitted twice: once for the accuracy suite, once for the Kappa suite.
            for mkeys, mtitles, mfile, mtitle, mprobe in metric_sets:
                if not any(cfg_summary(r, cfg).get(mprobe) is not None for r in ds_rs):
                    continue          # set not recorded in this run
                fig, axes = plt.subplots(4, 2, figsize=grid_size(4, 2, panel_aspect=0.62))
                fig.suptitle(f"{mtitle} -- {ds}{tag}", fontsize=FS_SUPTITLE, fontweight="bold")
                for ax, key, (title, defn) in zip(axes.flat, mkeys, mtitles):
                    dm = [(lab, cfg_summary(r, cfg).get(key), COLORS[i % len(COLORS)])
                          for i, (lab, r) in enumerate(zip(labels, ds_rs))
                          if cfg_summary(r, cfg).get(key) is not None]
                    ax.set_title(title, fontsize=FS_TITLE, pad=3)
                    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
                    if not dm:
                        ax.set_xticks([]); ax.set_ylim(-0.05, 0.05)
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes, fontsize=FS_TICK, color="#999999")
                        continue
                    sv = [d[1] for d in dm]; sc = [d[2] for d in dm]
                    xs = np.arange(len(dm))
                    ax.bar(xs, sv, color=sc, edgecolor="white", linewidth=0.5)
                    ax.set_xticks([])
                    ymin, ymax = label_bars(ax, xs, sv, colors=sc)
                    if "backward_transfer" in key:
                        ax.axhspan(0, ymax, alpha=0.07, color="green", zorder=0)
                        ax.axhspan(ymin, 0, alpha=0.07, color="red", zorder=0)
                for ax in axes.flat[len(mkeys):]:   # hide unused subplot(s)
                    ax.set_visible(False)
                model_legend(fig, labels, [COLORS[i % len(COLORS)] for i in range(len(labels))])
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                save_fig(fig, os.path.join(cdir, mfile))
                plt.close(fig)

            # 2) per-task final test accuracy
            xt = np.arange(n_tasks); width = 0.8 / max(len(ds_rs), 1)
            fig, ax = plt.subplots(figsize=fig_size(aspect=0.55))
            for i, r in enumerate(ds_rs):
                off = (i - len(ds_rs) / 2 + 0.5) * width
                ax.bar(xt + off, cfg_test(r, cfg), width, label=r["model"],
                       color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
            ax.set_xticks(xt); ax.set_xticklabels(task_names, fontsize=FS_TICK)
            ax.set_ylabel("Test Accuracy")
            ax.set_title(f"Per-Task Final Test Accuracy -- {ds}{tag}", fontweight="bold")
            ax.set_ylim(0, 1.32)
            ax.axhline(chance, color="gray", linewidth=0.6, linestyle="--", label="chance")
            ax.legend(fontsize=FS_LEGEND, ncol=3, loc="upper right")
            plt.tight_layout()
            save_fig(fig, os.path.join(cdir, "per_task.png"))
            plt.close(fig)

            seq = [r for r in ds_rs if len(cfg_R(r, cfg)) > 1]

            # 3a) per-task average accuracy
            if seq:
                w = 0.8 / len(seq)
                fig, ax = plt.subplots(figsize=fig_size(aspect=0.55))
                ax.set_title(f"Per-Task Average Accuracy -- {ds}{tag}", fontweight="bold")
                for i, r in enumerate(seq):
                    R = cfg_R(r, cfg); T = max(int(k) for k in R) + 1; avgs = []
                    for j in range(T):
                        col = [R.get(str(ii), {}).get(str(j)) for ii in range(j, T)]
                        col = [v for v in col if v is not None]
                        avgs.append(np.mean(col) if col else 0.0)
                    off = (i - len(seq) / 2 + 0.5) * w
                    ax.bar(np.arange(T) + off, avgs, w, label=r["model"],
                           color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
                ax.set_xticks(np.arange(n_tasks)); ax.set_xticklabels(task_names, fontsize=FS_TICK)
                ax.set_ylabel("Average Accuracy"); ax.set_ylim(0, 1.32)
                ax.axhline(chance, color="gray", linewidth=0.6, linestyle="--", label="chance")
                ax.legend(fontsize=FS_LEGEND, ncol=3, loc="upper right")
                plt.tight_layout()
                save_fig(fig, os.path.join(cdir, "per_task_avg_acc.png"))
                plt.close(fig)

            # 3b) per-task backward transfer
            if seq and n_tasks > 1:
                w = 0.8 / len(seq); xb = np.arange(n_tasks - 1); allb = []; per = []
                for r in seq:
                    R = cfg_R(r, cfg); T = max(int(k) for k in R) + 1; b = []
                    for j in range(T - 1):
                        rTj = R.get(str(T - 1), {}).get(str(j)); rjj = R.get(str(j), {}).get(str(j))
                        b.append((rTj - rjj) if (rTj is not None and rjj is not None) else 0.0)
                    per.append(b); allb.extend(b)
                ymin = min(min(allb) - 0.05, -0.05); ymax = max(max(allb) + 0.05, 0.05)
                fig, ax = plt.subplots(figsize=fig_size(aspect=0.55))
                ax.set_title(f"Per-Task Backward Transfer -- {ds}{tag}", fontweight="bold")
                ax.axhspan(0, ymax, alpha=0.07, color="green", zorder=0)
                ax.axhspan(ymin, 0, alpha=0.07, color="red", zorder=0)
                for i, (r, b) in enumerate(zip(seq, per)):
                    off = (i - len(seq) / 2 + 0.5) * w
                    ax.bar(xb[:len(b)] + off, b, w, label=r["model"],
                           color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
                ax.set_xticks(xb); ax.set_xticklabels(task_names[:n_tasks - 1], fontsize=FS_TICK)
                ax.set_ylabel("BWT"); ax.set_ylim(ymin, ymax)
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.legend(fontsize=FS_LEGEND, ncol=3, loc="upper right")
                plt.tight_layout()
                save_fig(fig, os.path.join(cdir, "per_task_bwt.png"))
                plt.close(fig)

            # 4) R-matrix and K-matrix heatmaps per model.
            #    Accuracy is a magnitude on [0,1] -> sequential ramp, light to dark.
            #    Kappa is a polarity around 0 (= chance agreement) on [-1,1] -> diverging
            #    ramp with a neutral midpoint, so below-chance cells are unmistakable
            #    instead of just "a bit lighter". Both scales are fixed rather than
            #    data-driven so panels stay comparable across models and configs.
            for getter, outdir, cmap, vmin, vmax, mname in (
                (cfg_R, rdir, "YlOrRd",  0.0, 1.0, "R"),
                (cfg_K, kdir, "RdBu_r", -1.0, 1.0, "K"),
            ):
                for r in ds_rs:
                    mat, n = dense(getter(r, cfg))
                    if mat is None:
                        continue
                    fig, ax = plt.subplots(figsize=fig_size(width_frac=0.7, aspect=0.9))
                    # Cells above the diagonal are tasks not yet seen. On the diverging
                    # ramp kappa=0 is white, so leaving them unpainted would make "not
                    # evaluated" look like "chance-level"; give them their own neutral.
                    cmap_obj = plt.get_cmap(cmap).copy()
                    cmap_obj.set_bad("#d0d0d0")
                    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap_obj, aspect="auto")
                    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cb.ax.tick_params(labelsize=FS_TICK)
                    if mname == "K":     # mark the chance line on the Kappa colourbar
                        cb.ax.axhline(0.0, color="#444444", linewidth=0.8)
                        cb.set_label("kappa (0 = chance)", fontsize=FS_TICK)
                    # White gridlines so adjacent same-colour cells don't merge.
                    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
                    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
                    ax.grid(which="minor", color="white", linewidth=1.0)
                    ax.tick_params(which="minor", length=0)
                    ax.set_xticks(range(n)); ax.set_yticks(range(n))
                    ax.set_xticklabels(task_names[:n], rotation=30, ha="right", fontsize=FS_TICK)
                    ax.set_yticklabels(task_names[:n], fontsize=FS_TICK)
                    ax.set_xlabel("Eval task"); ax.set_ylabel("After task")
                    ax.set_title(f"{mname}-matrix: {r['model']} -- {ds}{tag}", fontweight="bold")
                    for i in range(n):
                        for j in range(n):
                            v = mat[i, j]
                            if np.isnan(v):
                                continue
                            # both ends of the diverging ramp are dark; only the top end
                            # of the sequential one is.
                            dark = (v > 0.6) if mname == "R" else (abs(v) > 0.6)
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    fontsize=FS_ANNOT, color="white" if dark else "black")
                    slug_m = r["model"].lower().replace("-", "_")
                    plt.tight_layout()
                    save_fig(fig, os.path.join(outdir, f"{slug_m}.png"))
                    plt.close(fig)

        # Config comparison (no averaging) -- all configs side by side, per dataset.
        # x = models, one bar per config; one figure per metric. -> <dataset>/comparison/
        if len(cfgs) > 1:
            cmp_dir = os.path.join(plots_dir, ds_slug, "comparison")
            os.makedirs(cmp_dir, exist_ok=True)
            ccolors = config_colors(len(cfgs))
            clabel  = {c: " > ".join(pair_lbl(p) for p in ds_rs[0]["per_config"][c].get("task_pairs", []))
                       for c in cfgs}
            cwidth = 0.8 / len(cfgs)
            cmp_specs = [
                ("final_accuracy",           "Final Accuracy"),
                ("final_backward_transfer",  "Final BWT"),
                ("average_backward_transfer","Avg BWT"),
                ("final_cohen_kappa",        "Cohen's Kappa"),
                ("plasticity",               "Plasticity"),
                ("stability",                "Stability"),
                # Kappa counterparts
                ("average_kappa_over_time",         "Avg Kappa Over Time"),
                ("final_backward_transfer_kappa",   "Final BWT (Kappa)"),
                ("average_backward_transfer_kappa", "Avg BWT (Kappa)"),
                ("plasticity_kappa",                "Plasticity (Kappa)"),
                ("stability_kappa",                 "Stability (Kappa)"),
            ]
            for key, title in cmp_specs:
                sub = [r for r in ds_rs if r["per_config"][cfgs[0]]["summary"].get(key) is not None]
                if not sub:
                    continue
                xs = np.arange(len(sub))
                allv = [r["per_config"][c]["summary"].get(key) for c in cfgs for r in sub
                        if r["per_config"][c]["summary"].get(key) is not None]
                hi = max(allv + [0.0]); lo = min(allv + [0.0]); rng = (hi - lo) or 1.0
                fig, ax = plt.subplots(figsize=fig_size(aspect=0.62))
                for ci, c in enumerate(cfgs):
                    vals  = [r["per_config"][c]["summary"].get(key) for r in sub]
                    vplot = [v if v is not None else 0.0 for v in vals]
                    off   = (ci - len(cfgs) / 2 + 0.5) * cwidth
                    ax.bar(xs + off, vplot, cwidth, label=f"{c}: {clabel[c]}",
                           color=ccolors[ci], edgecolor="white", linewidth=0.4)
                    # per-bar value labels are illegible at this density (configs x models);
                    # exact values live in the summary table. Stub ~0 bars so colour shows.
                    zero_bar_stubs(ax, xs + off, [v if v is not None else None for v in vals],
                                   [ccolors[ci]] * len(vals), rng, width=cwidth)
                ax.set_xticks(xs)
                ax.set_xticklabels([r["model"] for r in sub], rotation=35, ha="right", fontsize=FS_TICK)
                ax.set_title(f"{ds} {title} -- config comparison", fontweight="bold")
                ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
                ax.set_ylim(lo - (0.08 * rng if lo < 0 else 0.02 * rng), hi + 0.08 * rng)
                # legend below the plot: a 15-config key placed inside would cover the bars.
                ax.legend(title="Config (task order)", fontsize=FS_LEGEND,
                          ncol=2 if len(cfgs) <= 10 else 3,
                          loc="upper center", bbox_to_anchor=(0.5, -0.22))
                save_fig(fig, os.path.join(cmp_dir, f"{key}.png"))
                plt.close(fig)

        # Aggregate over configs -- mean +/- std across all task configs, per dataset.
        # One bar per model with std error bars; complements (does not replace) the
        # per-config plots above. -> <dataset>/aggregate/metrics_mean_std.png
        if len(cfgs) > 1:
            agg_dir = os.path.join(plots_dir, ds_slug, "aggregate")
            os.makedirs(agg_dir, exist_ok=True)
            agg_labels = [r["model"] for r in ds_rs]
            # Cohen's Kappa is dropped from the accuracy aggregate figure (kept in the
            # per-config metrics.png and in the Kappa figure below). The Kappa set keeps
            # all of its panels. Each leaves one or two slots empty in the 4x2 grid.
            agg_sets = [
                ([(k, t) for k, t in zip(metric_keys, metric_titles)
                  if k != "final_cohen_kappa"],
                 "metrics_mean_std.png", "CL Metrics", "final_accuracy"),
                (list(zip(kappa_metric_keys, kappa_metric_titles)),
                 "metrics_kappa_mean_std.png", "CL Metrics (Cohen's Kappa)",
                 "average_kappa_over_time"),
            ]
            for agg_pairs, agg_file, agg_title, agg_probe in agg_sets:
                if not any(agg_metric(r, agg_probe)[0] is not None for r in ds_rs):
                    continue          # set not recorded in this run
                fig, axes = plt.subplots(4, 2, figsize=grid_size(4, 2, panel_aspect=0.62))
                fig.suptitle(f"{agg_title} (mean +/- std over {len(cfgs)} configs) -- {ds}",
                             fontsize=FS_SUPTITLE, fontweight="bold")
                for ax, (key, (title, defn)) in zip(axes.flat, agg_pairs):
                    dm = []
                    for i, (lab, r) in enumerate(zip(agg_labels, ds_rs)):
                        mu, sd = agg_metric(r, key)
                        if mu is not None:
                            dm.append((lab, mu, sd, COLORS[i % len(COLORS)]))
                    ax.set_title(title, fontsize=FS_TITLE, pad=3)
                    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
                    if not dm:
                        ax.set_xticks([]); ax.set_ylim(-0.05, 0.05)
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes, fontsize=FS_TICK, color="#999999")
                        continue
                    sv = [d[1] for d in dm]
                    se = [d[2] for d in dm]; sc = [d[3] for d in dm]
                    xs = np.arange(len(dm))
                    ax.bar(xs, sv, yerr=se, capsize=2, color=sc, edgecolor="white",
                           linewidth=0.5, error_kw={"elinewidth": 0.8})
                    ax.set_xticks([])
                    ymin, ymax = label_bars(ax, xs, sv, errs=se, colors=sc)
                    if "backward_transfer" in key:
                        ax.axhspan(0, ymax, alpha=0.07, color="green", zorder=0)
                        ax.axhspan(ymin, 0, alpha=0.07, color="red", zorder=0)
                for ax in axes.flat[len(agg_pairs):]:   # hide unused subplot(s)
                    ax.set_visible(False)
                model_legend(fig, agg_labels, [COLORS[i % len(COLORS)] for i in range(len(agg_labels))])
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                save_fig(fig, os.path.join(agg_dir, agg_file))
                plt.close(fig)

def run_main_experiment(model_dataset_params, datasets):
    """Phase 2a — main comparison: every model x dataset at REFERENCE_EPOCHS on all task
    configs. Writes summary.json + comparison plots + tables. Returns the experiment dir."""
    exp_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"experiment_{exp_ts}")
    os.makedirs(exp_dir, exist_ok=True)

    combos = [(ds, m) for ds in datasets for m in MODELS]
    # Each run reseeds the RNG itself (see run_combination), so sequential execution is
    # fully reproducible — every combination trains one after another on all CPU cores.
    print(f"\n{'='*70}\n  PHASE 2a — main comparison @ {REFERENCE_EPOCHS} epochs: "
          f"{len(combos)} combos, all configs (sequential)\n{'='*70}", flush=True)
    results_map = {}; _n = len(combos); _done = 0
    for ds, m in combos:
        results_map[(ds, m)] = run_combination(ds, m, model_dataset_params, exp_dir, REFERENCE_EPOCHS)
        _done += 1
        print(f"  >> Phase 2a progress: {_done}/{_n} combinations done "
              f"(latest: {m} on {ds})", flush=True)
    all_results = [results_map[(ds, m)] for ds, m in combos]
    save_results({
        "experiment":         exp_ts,
        "epochs":             REFERENCE_EPOCHS,
        "seed":               SEED,
        **subset_meta(datasets),
        "results":            all_results,
    }, os.path.join(exp_dir, "summary.json"))
    save_plots(all_results, exp_dir)
    print_metrics_table(all_results)
    print_hyperparams_table(all_results)
    return exp_dir


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Multi-Head (TIL) CL pipeline")
    _ap.add_argument("--datasets", nargs="+", default=DATASETS,
                     choices=["mnist", "wisdm", "hhar", "fashion_sw"],
                     help="Datasets to run (default: all). e.g. --datasets fashion_sw")
    _cli = _ap.parse_args()
    DATASETS = _cli.datasets

    _t_start = datetime.now()
    print(f"\n{'#'*70}\n#  MULTI-HEAD (TIL) EXPERIMENT\n{'#'*70}", flush=True)
    print(f"  started   : {_t_start:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  device    : {DEVICE}", flush=True)
    print(f"  datasets  : {DATASETS}", flush=True)
    print(f"  models    : {len(MODELS)}  ->  {MODELS}", flush=True)
    print(f"  total Phase-2 combinations: {len(DATASETS) * len(MODELS)}", flush=True)

    # Phase 1: best params from the shared tuning cache.
    model_dataset_params = load_tuned_params(TUNED_PARAMS_CACHE, DATASETS, MODELS)

    # Phase 2a: main comparison.
    exp_dir = run_main_experiment(model_dataset_params, DATASETS)

    _elapsed = datetime.now() - _t_start
    print(f"\n{'#'*70}\n#  DONE — multi-head experiment complete\n{'#'*70}", flush=True)
    print(f"  elapsed   : {_elapsed}", flush=True)
    print(f"  results   : {exp_dir}", flush=True)
