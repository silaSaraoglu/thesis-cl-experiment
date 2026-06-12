"""
ESN Naive (fine-tuning) -- Sequential MNIST.

CL strategy: Naive — plain gradient descent with no anti-forgetting mechanism.
  The readout is updated on each new task without constraint.
  This is the LOWER BOUND: the model will catastrophically forget earlier tasks.

Avalanche Naive strategy = standard cross-entropy training, one epoch at a time,
no regularisation and no memory buffer.

Original repo: models/ESN/repo/experiments/splitmnist_esn.py  (strategy='naive')

Tasks: 5 binary classification tasks
  Task 1: digit 0 vs 1
  Task 2: digit 2 vs 3
  Task 3: digit 4 vs 5
  Task 4: digit 6 vs 7
  Task 5: digit 8 vs 9
Input: MNIST as row-based sequence (B, 28, 28) — 28 time steps × 28 pixel features.
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

from shared.dataset_cl import MNIST_CL
from shared.metrics import CLMetrics, cohen_kappa
from models.ESN.esn_utils import SingleHeadESN
from shared.utils import collect_datasets
from clrnn.utils import get_strategy           # original ESN repo utility
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.plugins import EvaluationPlugin

_TT = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]  # label pairs per task
_NUM_TASKS = 5

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

def run_esn_naive_mnist(args, verbose = True, trial=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device = args.device
    data_root = args.data_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    image_size = 28
    max_samples = args.subset
    task_pairs = getattr(args, 'task_pairs', _TT)

    mnist_train = MNIST_CL(data_root, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=2)
    mnist_test = MNIST_CL(data_root, download=False, train=False,
                           output_size=2)
    mnist_train.set_holdout_config(
        holdout_n = getattr(args, "holdout_n", 0),
        holdout_seed = getattr(args, "holdout_seed", 0),
        use_holdout = getattr(args, "use_holdout", False),
    )


    train_datasets, val_datasets, test_datasets = collect_datasets(
        mnist_train, mnist_test, task_pairs, max_samples, batch_size, reshape=True)
    scenario = dataset_benchmark(train_datasets, test_datasets)

    model = SingleHeadESN(input_size=image_size, args=args)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    args_cl = Namespace(**{**vars(args), "strategy": "naive"})
    cl_strategy = get_strategy(model, opt, criterion, EvaluationPlugin(), device, args_cl)

    metrics = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id, exp in enumerate(scenario.train_stream):

        loader_pt = DataLoader(test_datasets[task_id], batch_size=batch_size, shuffle=False)
        # pre-train performance is recorded for the forward transfer metric
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
            acc = float(num_of_correct_predictions / num_of_samples)
            kappa = cohen_kappa(all_labels, all_predictions)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"    task {eval_id+1} [{task_pairs[eval_id][0]}/{task_pairs[eval_id][1]}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"  MNIST results  --  ESN-Naive")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
