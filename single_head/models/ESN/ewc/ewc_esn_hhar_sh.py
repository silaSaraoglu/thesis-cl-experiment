"""
ESN with EWC -- HHAR  (Domain-Incremental).

CL strategy: Elastic Weight Consolidation (separate Fisher per task) on the readout.
DIL: one shared 6-way head; tasks = device models. Input: (B, 128, 3).
"""
import os, sys, warnings
import torch
import torch.nn as nn
import optuna
import numpy as np
from torch.utils.data import DataLoader
from argparse import Namespace

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import HHAR_CL
from shared.metrics import CLMetrics, cohen_kappa
from models.ESN.esn_utils import SingleHeadESN
from shared.utils import collect_datasets
from clrnn.utils import get_strategy
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.plugins import EvaluationPlugin

_TASKS     = [0, 1, 2, 3]
_NUM_TASKS = 4

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total = 0
    for x, y in loader:
        preds = model.predict(x)
        labels = y.numpy()
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                num_correct += 1
        num_total += y.size(0)
    return num_correct / num_total

def run_esn_ewc_hhar(args, verbose = True, trial=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device        = args.device
    data_root     = args.data_dir
    batch_size    = args.batch_size
    learning_rate = args.learning_rate
    max_samples   = args.subset
    task_order    = getattr(args, 'task_pairs', _TASKS)

    hhar_train = HHAR_CL(data_root, train=True,  download=False, perc_val=0.25, batch_size=batch_size)
    hhar_test  = HHAR_CL(data_root, train=False, download=False)
    hhar_train.set_holdout_config(
        holdout_n    = getattr(args, "holdout_n", 0),
        holdout_seed = getattr(args, "holdout_seed", 0),
        use_holdout  = getattr(args, "use_holdout", False),
    )

    train_datasets, val_datasets, test_datasets = collect_datasets(
        hhar_train, hhar_test, task_order, max_samples, batch_size)
    scenario = dataset_benchmark(train_datasets, test_datasets)

    model     = SingleHeadESN(input_size=3, args=args, num_classes=6)
    opt       = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    args_cl = Namespace(**{**vars(args), "strategy": "ewc",
                           "ewc_mode": "separate"})
    cl_strategy = get_strategy(model, opt, criterion, EvaluationPlugin(), device, args_cl)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id, exp in enumerate(scenario.train_stream):

        loader_pt = DataLoader(test_datasets[task_id], batch_size=batch_size, shuffle=False)
        metrics.record_pretrain(task_id, calculate_accuracy(model, loader_pt))

        cl_strategy.train(exp)

        loader_val = DataLoader(val_datasets[task_id], batch_size=batch_size, shuffle=False)
        subtask_val_accs.append(calculate_accuracy(model, loader_val))
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            loader_te = DataLoader(test_datasets[eval_id], batch_size=batch_size, shuffle=False)
            all_predictions, all_labels = [], []
            for x, y in loader_te:
                all_predictions.extend(model.predict(x))
                all_labels.extend(y.numpy())
            num_of_correct_predictions = 0
            num_of_samples = len(all_labels)
            for i in range(num_of_samples):
                if all_predictions[i] == all_labels[i]:
                    num_of_correct_predictions += 1
            acc   = float(num_of_correct_predictions / num_of_samples)
            kappa = cohen_kappa(all_labels, all_predictions)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            eval_name = HHAR_CL.DEVICE_MODELS[task_order[eval_id]]
            if verbose:
                print(f"    task {eval_id+1} [{eval_name}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"  HHAR results  --  ESN-EWC")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
