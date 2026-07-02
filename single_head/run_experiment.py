import warnings
warnings.filterwarnings("ignore")  # silence import-time lib warnings (jaxopt/Avalanche/torch)
                                   # before torch & the model modules are imported below
import sys
import os
import numpy as np
import torch
import argparse
from argparse import Namespace
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
_p = HERE
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
sys.path.insert(0, HERE)

from models.GIM_LSTM.gim_lstm_mnist_sh import run_gim_mnist
from models.LSTM.naive.naive_lstm_mnist_sh import run_naive_lstm_mnist_sh
from models.LSTM.joint.joint_lstm_mnist_sh import run_joint_lstm_mnist_sh
from models.LSTM.replay.replay_lstm_mnist_sh import run_lstm_replay_mnist_sh
from models.ESN.naive.naive_esn_mnist_sh import run_esn_naive_mnist
from models.ESN.joint.joint_esn_mnist_sh import run_joint_esn_mnist_sh
from models.ESN.ewc.ewc_esn_mnist_sh import run_esn_ewc_mnist
from models.ESN.lwf.lwf_esn_mnist_sh import run_esn_lwf_mnist
from models.ESN.replay.replay_esn_mnist_sh import run_esn_replay_mnist
from models.ESN.slda.slda_esn_mnist_sh import run_esn_slda_mnist
from models.GIM_LSTM.gim_lstm_hhar_sh import run_gim_hhar
from models.LSTM.naive.naive_lstm_hhar_sh import run_naive_lstm_hhar_sh
from models.LSTM.joint.joint_lstm_hhar_sh import run_joint_lstm_hhar_sh
from models.LSTM.replay.replay_lstm_hhar_sh import run_lstm_replay_hhar_sh
from models.ESN.naive.naive_esn_hhar_sh import run_esn_naive_hhar
from models.ESN.joint.joint_esn_hhar_sh import run_joint_esn_hhar_sh
from models.ESN.ewc.ewc_esn_hhar_sh import run_esn_ewc_hhar
from models.ESN.lwf.lwf_esn_hhar_sh import run_esn_lwf_hhar
from models.ESN.replay.replay_esn_hhar_sh import run_esn_replay_hhar
from models.ESN.slda.slda_esn_hhar_sh import run_esn_slda_hhar
from models.GIM_LSTM.gim_lstm_fashionsw_sh import run_gim_fashionsw
from models.LSTM.naive.naive_lstm_fashionsw_sh import run_naive_lstm_fashionsw_sh
from models.LSTM.joint.joint_lstm_fashionsw_sh import run_joint_lstm_fashionsw_sh
from models.LSTM.replay.replay_lstm_fashionsw_sh import run_lstm_replay_fashionsw_sh
from models.ESN.naive.naive_esn_fashionsw_sh import run_esn_naive_fashionsw
from models.ESN.joint.joint_esn_fashionsw_sh import run_joint_esn_fashionsw_sh
from models.ESN.ewc.ewc_esn_fashionsw_sh import run_esn_ewc_fashionsw
from models.ESN.lwf.lwf_esn_fashionsw_sh import run_esn_lwf_fashionsw
from models.ESN.replay.replay_esn_fashionsw_sh import run_esn_replay_fashionsw
from models.ESN.slda.slda_esn_fashionsw_sh import run_esn_slda_fashionsw
from shared.metrics import save_results
from shared.utils import load_tuned_params


from config import *

TASK_META = {
    "mnist":      {"num_tasks": 5},
    "hhar":       {"num_tasks": 4, "task_names": {0:"nexus4",1:"s3",2:"s3mini",3:"samsungold"}},
    "fashion_sw": {"num_tasks": 3},
}

DATASET_LABEL = {"mnist": "MNIST", "hhar": "HHAR", "fashion_sw": "Fashion-SW"}

RUNNERS = {
    ("mnist",   "gim_alstm_sh"):    lambda a, v: run_gim_mnist("alstm", a, v),
    ("mnist",   "lstm_sh"):         lambda a, v: run_naive_lstm_mnist_sh(a, v),
    ("mnist",   "joint_lstm_sh"):   lambda a, v: run_joint_lstm_mnist_sh(a, v),
    ("mnist",   "lstm_replay_sh"):  lambda a, v: run_lstm_replay_mnist_sh(a, v),
    ("mnist",   "esn_naive_sh"):    lambda a, v: run_esn_naive_mnist(a, v),
    ("mnist",   "esn_ewc_sh"):      lambda a, v: run_esn_ewc_mnist(a, v),
    ("mnist",   "esn_lwf_sh"):      lambda a, v: run_esn_lwf_mnist(a, v),
    ("mnist",   "esn_replay_sh"):   lambda a, v: run_esn_replay_mnist(a, v),
    ("mnist",   "esn_slda_sh"):     lambda a, v: run_esn_slda_mnist(a, v),
    ("mnist",   "joint_esn_sh"):    lambda a, v: run_joint_esn_mnist_sh(a, v),
    ("hhar",      "gim_alstm_sh"):    lambda a, v: run_gim_hhar("alstm", a, v),
    ("hhar",      "lstm_sh"):         lambda a, v: run_naive_lstm_hhar_sh(a, v),
    ("hhar",      "joint_lstm_sh"):   lambda a, v: run_joint_lstm_hhar_sh(a, v),
    ("hhar",      "lstm_replay_sh"):  lambda a, v: run_lstm_replay_hhar_sh(a, v),
    ("hhar",      "esn_naive_sh"):    lambda a, v: run_esn_naive_hhar(a, v),
    ("hhar",      "esn_ewc_sh"):      lambda a, v: run_esn_ewc_hhar(a, v),
    ("hhar",      "esn_lwf_sh"):      lambda a, v: run_esn_lwf_hhar(a, v),
    ("hhar",      "esn_replay_sh"):   lambda a, v: run_esn_replay_hhar(a, v),
    ("hhar",      "esn_slda_sh"):     lambda a, v: run_esn_slda_hhar(a, v),
    ("hhar",      "joint_esn_sh"):    lambda a, v: run_joint_esn_hhar_sh(a, v),
    ("fashion_sw","gim_alstm_sh"):    lambda a, v: run_gim_fashionsw("alstm", a, v),
    ("fashion_sw","lstm_sh"):         lambda a, v: run_naive_lstm_fashionsw_sh(a, v),
    ("fashion_sw","joint_lstm_sh"):   lambda a, v: run_joint_lstm_fashionsw_sh(a, v),
    ("fashion_sw","lstm_replay_sh"):  lambda a, v: run_lstm_replay_fashionsw_sh(a, v),
    ("fashion_sw","esn_naive_sh"):    lambda a, v: run_esn_naive_fashionsw(a, v),
    ("fashion_sw","esn_ewc_sh"):      lambda a, v: run_esn_ewc_fashionsw(a, v),
    ("fashion_sw","esn_lwf_sh"):      lambda a, v: run_esn_lwf_fashionsw(a, v),
    ("fashion_sw","esn_replay_sh"):   lambda a, v: run_esn_replay_fashionsw(a, v),
    ("fashion_sw","esn_slda_sh"):     lambda a, v: run_esn_slda_fashionsw(a, v),
    ("fashion_sw","joint_esn_sh"):    lambda a, v: run_joint_esn_fashionsw_sh(a, v),
}

SKIP_KEYS = {"results_dir", "subset", "data_dir", "device",
             "holdout_n", "holdout_seed", "use_holdout"}

def build_args(model, dataset, model_dataset_params):
    args = Namespace(**model_dataset_params[(model, dataset)])
    args.results_dir = RESULTS_DIR
    if dataset == "mnist":
        args.data_dir    = MNIST_DIR
        args.subset      = SUBSET_MNIST
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



def disp_pairs(pairs, task_names):
    # HHAR configs are scalar device ids -> render the device name for plots/tables;
    # class-pair configs (e.g. [0,1]) are already meaningful and pass through unchanged.
    return [task_names.get(p, p) if not isinstance(p, (list, tuple)) else p for p in pairs]

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

def cfg_test(r, cfg):
    return r["per_config"][cfg]["test"]

def cfg_pretrain(r, cfg):
    return r["per_config"][cfg].get("pretrain", {})

def agg_metric(r, key):
    pc = r.get("per_config") or {}
    vals = [c["summary"].get(key) for c in pc.values()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)), float(np.std(vals))

def fig_footer(fig, all_results):
    # Report subsets only for the datasets actually present in this figure's results.
    ds_ids = [d for d in DATASETS if DATASET_LABEL.get(d) in {r["dataset"] for r in all_results}]
    if not ds_ids:
        return
    train = ", ".join(f"{DATASET_LABEL[d]}={SUBSET_BY_DATASET[d] if SUBSET_BY_DATASET[d] is not None else 'full'}"
                      for d in ds_ids)
    tune  = ", ".join(f"{DATASET_LABEL[d]}={TUNE_SUBSET_BY_DATASET[d]}" for d in ds_ids)
    fig.text(0.5, 0.002, f"Train subset: {train}  |  Tune subset: {tune}",
             ha="center", va="bottom", fontsize=7, color="#777777", style="italic")


def run_combination(dataset, model, model_dataset_params,
                    exp_dir, epochs, config_names=None):
    # Runs are executed sequentially (one combination at a time), so let torch use all
    # available CPU cores for each run instead of capping it to a single thread.
    meta          = TASK_META[dataset]
    task_names    = meta.get("task_names", {})
    runner        = RUNNERS[(dataset, model)]

    args          = build_args(model, dataset, model_dataset_params)
    args.epochs = epochs
    model_label   = model.upper().replace("_", "-")
    dataset_label = DATASET_LABEL[dataset]

    hyperparams = {k: v for k, v in vars(args).items() if k not in SKIP_KEYS}

    chance = 1.0 / 6 if dataset == "hhar" else 0.5
    task_cfgs = DATASET_CONFIGS[dataset]
    cfg_names = config_names if config_names is not None else list(task_cfgs)
    per_config = {}
    for cfg_name in cfg_names:
        task_pairs = task_cfgs[cfg_name]
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        run_args = Namespace(**vars(args), task_pairs=task_pairs)
        result   = runner(run_args, False)
        per_config[cfg_name] = {
            "val":     [float(v) for v in result[0]],
            "test":    [float(v) for v in result[1]],
            "summary": result[2].summary(chance),
            "R_matrix": {str(i): {str(j): v for j, v in d.items()}
                         for i, d in result[2].R.items()},
            "pretrain": {str(k): float(v) for k, v in result[2].pretrain.items()},
            "task_pairs": disp_pairs(task_pairs, task_names),
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

def print_metrics_table(all_results):
    cols = ["Model", "Dataset", "Acc_final", "Acc_avg", "BWT_final", "BWT_avg",
            "FWT", "Kappa", "Plasticity", "Stability"]
    col_widths = [15, 12, 10, 10, 10, 10, 8, 8, 11, 10]
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
                format_metric(s.get('forward_transfer')),          format_metric(s.get('final_cohen_kappa')),
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
                    "forward_transfer", "final_cohen_kappa", "plasticity", "stability"]
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

    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metric_keys = [
        "final_accuracy", "average_accuracy_over_time",
        "final_backward_transfer", "average_backward_transfer",
        "forward_transfer", "final_cohen_kappa", "plasticity", "stability",
    ]
    metric_titles = [
        ("Final Accuracy",         "mean R[T-1][j] over all j"),
        ("Avg Accuracy Over Time", "mean R[i][j] for all i>=j"),
        ("Final BWT",              "mean(R[T-1][j]-R[j][j]) for j<T-1"),
        ("Avg BWT",                "mean(R[i][j]-R[j][j]) for all i>j"),
        ("Forward Transfer",       "mean(pretrain(j)-chance) for j>0"),
        ("Cohen's Kappa",          "agreement above chance"),
        ("Plasticity",             "mean diagonal R[j][j]"),
        ("Stability",              "mean off-diag (i>j)"),
    ]

    for ds in sorted(set(r["dataset"] for r in all_results)):
        ds_rs   = [r for r in all_results if r["dataset"] == ds]
        ds_slug = ds.lower().replace("-", "_").replace(" ", "")
        chance  = 1.0 / 6 if "hhar" in ds.lower() else 0.5   # 1/num_classes: FWT baseline + plot chance line
        cfgs    = list(ds_rs[0]["per_config"].keys())

        for cfg in cfgs:
            cdir  = os.path.join(plots_dir, ds_slug, cfg)
            rdir  = os.path.join(cdir, "rmatrix")
            os.makedirs(rdir, exist_ok=True)

            task_names = [pair_lbl(p) for p in ds_rs[0]["per_config"][cfg].get("task_pairs", [])]
            n_tasks = len(task_names)
            labels  = [r["model"] for r in ds_rs]
            tag     = f"  [{cfg}]"

            # 1) CL-metrics overview (only models that define each metric)
            fig, axes = plt.subplots(2, 4, figsize=(20, 8))
            fig.suptitle(f"CL Metrics -- {ds}{tag}", fontsize=12, fontweight="bold")
            for ax, key, (title, defn) in zip(axes.flat, metric_keys, metric_titles):
                dm = [(lab, cfg_summary(r, cfg).get(key), COLORS[i % len(COLORS)])
                      for i, (lab, r) in enumerate(zip(labels, ds_rs))
                      if cfg_summary(r, cfg).get(key) is not None]
                ax.set_title(f"{title}\n[{defn}]", fontsize=8, pad=4)
                ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
                if not dm:
                    ax.set_xticks([]); ax.set_ylim(-0.05, 0.05)
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="#999999")
                    continue
                sl = [d[0] for d in dm]; sv = [d[1] for d in dm]; sc = [d[2] for d in dm]
                xs = np.arange(len(dm))
                bars = ax.bar(xs, sv, color=sc, edgecolor="white", linewidth=0.5)
                ax.set_xticks(xs); ax.set_xticklabels(sl, rotation=40, ha="right", fontsize=7)
                ymin = min(min(sv) - 0.05, -0.05); ymax = max(max(sv) + 0.05, 0.05)
                ax.set_ylim(ymin, ymax)
                if key in ("final_backward_transfer", "average_backward_transfer", "forward_transfer"):
                    ax.axhspan(0, ymax, alpha=0.07, color="green", zorder=0)
                    ax.axhspan(ymin, 0, alpha=0.07, color="red", zorder=0)
                for bar, v in zip(bars, sv):
                    va = "bottom" if v >= 0 else "top"; off = 0.01 if v >= 0 else -0.02
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off,
                            f"{v:.2f}", ha="center", va=va, fontsize=6)
            plt.tight_layout(); fig_footer(fig, all_results)
            fig.savefig(os.path.join(cdir, "metrics.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 2) per-task final test accuracy
            xt = np.arange(n_tasks); width = 0.8 / max(len(ds_rs), 1)
            fig, ax = plt.subplots(figsize=(max(8, n_tasks * len(ds_rs) * 0.35 + 2), 5))
            for i, r in enumerate(ds_rs):
                off = (i - len(ds_rs) / 2 + 0.5) * width
                ax.bar(xt + off, cfg_test(r, cfg), width, label=r["model"],
                       color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
            ax.set_xticks(xt); ax.set_xticklabels(task_names, fontsize=8)
            ax.set_ylabel("Test Accuracy")
            ax.set_title(f"Per-Task Final Test Accuracy -- {ds}{tag}\n[R[T-1][j]]", fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.axhline(chance, color="gray", linewidth=0.6, linestyle="--", label="chance")
            ax.legend(fontsize=7, ncol=3, loc="upper right")
            plt.tight_layout(); fig_footer(fig, all_results)
            fig.savefig(os.path.join(cdir, "per_task.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            seq = [r for r in ds_rs if len(cfg_R(r, cfg)) > 1]

            # 3a) per-task average accuracy
            if seq:
                w = 0.8 / len(seq)
                fig, ax = plt.subplots(figsize=(max(8, n_tasks * len(seq) * 0.35 + 2), 5))
                ax.set_title(f"Per-Task Average Accuracy -- {ds}{tag}\n"
                             "[avg_acc_j = mean(R[i][j] for i>=j)]", fontweight="bold")
                for i, r in enumerate(seq):
                    R = cfg_R(r, cfg); T = max(int(k) for k in R) + 1; avgs = []
                    for j in range(T):
                        col = [R.get(str(ii), {}).get(str(j)) for ii in range(j, T)]
                        col = [v for v in col if v is not None]
                        avgs.append(np.mean(col) if col else 0.0)
                    off = (i - len(seq) / 2 + 0.5) * w
                    ax.bar(np.arange(T) + off, avgs, w, label=r["model"],
                           color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
                ax.set_xticks(np.arange(n_tasks)); ax.set_xticklabels(task_names, fontsize=8)
                ax.set_ylabel("Average Accuracy"); ax.set_ylim(0, 1.05)
                ax.axhline(chance, color="gray", linewidth=0.6, linestyle="--", label="chance")
                ax.legend(fontsize=7, ncol=3, loc="upper right")
                plt.tight_layout(); fig_footer(fig, all_results)
                fig.savefig(os.path.join(cdir, "per_task_avg_acc.png"), dpi=150, bbox_inches="tight")
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
                fig, ax = plt.subplots(figsize=(max(8, (n_tasks - 1) * len(seq) * 0.35 + 2), 5))
                ax.set_title(f"Per-Task Backward Transfer -- {ds}{tag}\n"
                             "[BWT_j = R[T-1][j]-R[j][j]]", fontweight="bold")
                ax.axhspan(0, ymax, alpha=0.07, color="green", zorder=0)
                ax.axhspan(ymin, 0, alpha=0.07, color="red", zorder=0)
                for i, (r, b) in enumerate(zip(seq, per)):
                    off = (i - len(seq) / 2 + 0.5) * w
                    ax.bar(xb[:len(b)] + off, b, w, label=r["model"],
                           color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
                ax.set_xticks(xb); ax.set_xticklabels(task_names[:n_tasks - 1], fontsize=8)
                ax.set_ylabel("BWT"); ax.set_ylim(ymin, ymax)
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.legend(fontsize=7, ncol=3, loc="upper right")
                plt.tight_layout(); fig_footer(fig, all_results)
                fig.savefig(os.path.join(cdir, "per_task_bwt.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

            # 3c) per-task forward transfer
            if seq and n_tasks > 1:
                w = 0.8 / len(seq); xf = np.arange(n_tasks - 1); allf = []; per = []
                for r in seq:
                    # FWT_j = (zero-shot acc on task j, before training it) - chance.
                    # The pre-task accuracy lives in the serialized `pretrain` dict.
                    P = cfg_pretrain(r, cfg); fwt = []
                    for j in range(1, n_tasks):
                        pj = P.get(str(j))
                        fwt.append((pj - chance) if pj is not None else 0.0)
                    per.append(fwt); allf.extend(fwt)
                ymin = min(min(allf) - 0.05, -0.05); ymax = max(max(allf) + 0.05, 0.05)
                fig, ax = plt.subplots(figsize=(max(8, (n_tasks - 1) * len(seq) * 0.35 + 2), 5))
                ax.set_title(f"Per-Task Forward Transfer -- {ds}{tag}\n"
                             "[FWT_j = pretrain_acc(j) - chance]", fontweight="bold")
                ax.axhspan(0, ymax, alpha=0.07, color="green", zorder=0)
                ax.axhspan(ymin, 0, alpha=0.07, color="red", zorder=0)
                for i, (r, fwt) in enumerate(zip(seq, per)):
                    off = (i - len(seq) / 2 + 0.5) * w
                    ax.bar(xf[:len(fwt)] + off, fwt, w, label=r["model"],
                           color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
                ax.set_xticks(xf); ax.set_xticklabels(task_names[1:], fontsize=8)
                ax.set_ylabel("FWT"); ax.set_ylim(ymin, ymax)
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.legend(fontsize=7, ncol=3, loc="upper right")
                plt.tight_layout(); fig_footer(fig, all_results)
                fig.savefig(os.path.join(cdir, "per_task_fwt.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

            # 4) R-matrix heatmap per model
            for r in ds_rs:
                mat, n = dense(cfg_R(r, cfg))
                if mat is None:
                    continue
                fig, ax = plt.subplots(figsize=(max(4, n + 1), max(4, n + 1)))
                im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(task_names[:n], rotation=30, ha="right", fontsize=8)
                ax.set_yticklabels(task_names[:n], fontsize=8)
                ax.set_xlabel("Eval task"); ax.set_ylabel("After task")
                ax.set_title(f"R-matrix: {r['model']} -- {ds}{tag}", fontweight="bold")
                for i in range(n):
                    for j in range(n):
                        if not np.isnan(mat[i, j]):
                            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                                    fontsize=8, color="black" if mat[i, j] > 0.4 else "white")
                slug_m = r["model"].lower().replace("-", "_")
                plt.tight_layout()
                fig.savefig(os.path.join(rdir, f"{slug_m}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

        # Config comparison (no averaging) -- all configs side by side, per dataset.
        # x = models, one bar per config; one figure per metric. -> <dataset>/comparison/
        if len(cfgs) > 1:
            cmp_dir = os.path.join(plots_dir, ds_slug, "comparison")
            os.makedirs(cmp_dir, exist_ok=True)
            ccolors = [COLORS[i % len(COLORS)] for i in range(len(cfgs))]
            clabel  = {c: " > ".join(pair_lbl(p) for p in ds_rs[0]["per_config"][c].get("task_pairs", []))
                       for c in cfgs}
            cwidth = 0.8 / len(cfgs)
            cmp_specs = [
                ("final_accuracy",           "Final Accuracy"),
                ("final_backward_transfer",  "Final BWT"),
                ("average_backward_transfer","Avg BWT"),
                ("forward_transfer",         "Forward Transfer"),
                ("final_cohen_kappa",        "Cohen's Kappa"),
                ("plasticity",               "Plasticity"),
                ("stability",                "Stability"),
            ]
            for key, title in cmp_specs:
                sub = [r for r in ds_rs if r["per_config"][cfgs[0]]["summary"].get(key) is not None]
                if not sub:
                    continue
                xs = np.arange(len(sub))
                fig, ax = plt.subplots(figsize=(max(8, len(sub) * len(cfgs) * 0.45 + 2), 5))
                for ci, c in enumerate(cfgs):
                    vals  = [r["per_config"][c]["summary"].get(key) for r in sub]
                    vplot = [v if v is not None else 0.0 for v in vals]
                    off   = (ci - len(cfgs) / 2 + 0.5) * cwidth
                    bars = ax.bar(xs + off, vplot, cwidth, label=f"{c}: {clabel[c]}",
                                  color=ccolors[ci], edgecolor="white", linewidth=0.4)
                    for bar, v in zip(bars, vals):
                        if v is None:
                            bar.set_visible(False); continue
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f"{v:.2f}", ha="center", va="bottom", fontsize=6)
                ax.set_xticks(xs)
                ax.set_xticklabels([r["model"] for r in sub], rotation=35, ha="right", fontsize=8)
                ax.set_title(f"{ds} {title} -- config comparison", fontweight="bold")
                ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
                ax.set_ylim(min(ax.get_ylim()[0], -0.05), max(ax.get_ylim()[1], 0.05))
                ax.legend(title="Config (task order)", fontsize=7)
                plt.tight_layout(); fig_footer(fig, all_results)
                fig.savefig(os.path.join(cmp_dir, f"{key}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

        if len(cfgs) > 1:
            agg_dir = os.path.join(plots_dir, ds_slug, "aggregate")
            os.makedirs(agg_dir, exist_ok=True)
            agg_labels = [r["model"] for r in ds_rs]
            fig, axes = plt.subplots(2, 4, figsize=(20, 8))
            fig.suptitle(f"CL Metrics (mean +/- std over {len(cfgs)} configs) -- {ds}",
                         fontsize=12, fontweight="bold")
            for ax, key, (title, defn) in zip(axes.flat, metric_keys, metric_titles):
                dm = []
                for i, (lab, r) in enumerate(zip(agg_labels, ds_rs)):
                    mu, sd = agg_metric(r, key)
                    if mu is not None:
                        dm.append((lab, mu, sd, COLORS[i % len(COLORS)]))
                ax.set_title(f"{title}\n[{defn}]", fontsize=8, pad=4)
                ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
                if not dm:
                    ax.set_xticks([]); ax.set_ylim(-0.05, 0.05)
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="#999999")
                    continue
                sl = [d[0] for d in dm]; sv = [d[1] for d in dm]
                se = [d[2] for d in dm]; sc = [d[3] for d in dm]
                xs = np.arange(len(dm))
                ax.bar(xs, sv, yerr=se, capsize=3, color=sc, edgecolor="white",
                       linewidth=0.5, error_kw={"elinewidth": 0.8})
                ax.set_xticks(xs); ax.set_xticklabels(sl, rotation=40, ha="right", fontsize=7)
                ymin = min(min(np.subtract(sv, se)) - 0.05, -0.05)
                ymax = max(max(np.add(sv, se)) + 0.05, 0.05)
                ax.set_ylim(ymin, ymax)
                if key in ("final_backward_transfer", "average_backward_transfer", "forward_transfer"):
                    ax.axhspan(0, ymax, alpha=0.07, color="green", zorder=0)
                    ax.axhspan(ymin, 0, alpha=0.07, color="red", zorder=0)
                for x, v, e in zip(xs, sv, se):
                    va = "bottom" if v >= 0 else "top"
                    off = (e + 0.01) if v >= 0 else -(e + 0.02)
                    ax.text(x, v + off, f"{v:.2f}", ha="center", va=va, fontsize=6)
            plt.tight_layout(); fig_footer(fig, all_results)
            fig.savefig(os.path.join(agg_dir, "metrics_mean_std.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

def run_main_experiment(model_dataset_params, datasets):
    """Phase 2a — main comparison: every model x dataset at EPOCHS on all task
    configs. Writes summary.json + comparison plots + tables. Returns the experiment dir."""
    exp_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"experiment_{exp_ts}")
    os.makedirs(exp_dir, exist_ok=True)

    combos = [(ds, m) for ds in datasets for m in MODELS]
    results_map = {}
    for ds, m in combos:
        results_map[(ds, m)] = run_combination(ds, m, model_dataset_params, exp_dir, EPOCHS)
    all_results = [results_map[(ds, m)] for ds, m in combos]
    save_results({
        "experiment":        exp_ts,
        "epochs":            EPOCHS,
        "seed":              SEED,
        "datasets":          datasets,
        "subsets":           {d: SUBSET_BY_DATASET[d] for d in datasets},
        "tune_holdouts":     {d: TUNE_SUBSET_BY_DATASET[d] for d in datasets},
        "tune_split_seed":   TUNE_SPLIT_SEED,
        "results":           all_results,
    }, os.path.join(exp_dir, "summary.json"))
    save_plots(all_results, exp_dir)
    print_metrics_table(all_results)
    print_hyperparams_table(all_results)
    return exp_dir


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Single-Head (DIL) CL pipeline")
    _ap.add_argument("--datasets", nargs="+", default=DATASETS,
                     choices=["mnist", "hhar", "fashion_sw"])
    _cli = _ap.parse_args()
    DATASETS = _cli.datasets

    _t_start = datetime.now()
    print(f"\n{'#'*70}\n#  SINGLE-HEAD (DIL) EXPERIMENT\n{'#'*70}", flush=True)
    print(f"  started   : {_t_start:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  device    : {DEVICE}", flush=True)
    print(f"  datasets  : {DATASETS}", flush=True)
    print(f"  models    : {len(MODELS)}  ->  {MODELS}", flush=True)
    print(f"  total Phase-2 combinations: {len(DATASETS) * len(MODELS)}", flush=True)

    # Phase 1: best-params from the shared cache (tunes only what's missing, then reuses it).
    model_dataset_params = load_tuned_params(TUNED_PARAMS_CACHE, DATASETS, MODELS, SEED)

    # Phase 2a: main comparison (the epoch sweep lives in run_epoch_analysis.py).
    exp_dir = run_main_experiment(model_dataset_params, DATASETS)

    _elapsed = datetime.now() - _t_start
    print(f"\n{'#'*70}\n#  DONE — single-head experiment complete\n{'#'*70}", flush=True)
    print(f"  elapsed   : {_elapsed}", flush=True)
    print(f"  results   : {exp_dir}", flush=True)
