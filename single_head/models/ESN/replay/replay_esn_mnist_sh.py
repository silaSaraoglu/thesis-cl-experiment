"""
ESN with Experience Replay -- Sequential MNIST.

CL strategy: Experience Replay (rehearsal memory).
  A fixed-size ring buffer stores mem_size samples from past experiences.
  Each mini-batch during new-task training is augmented with replayed samples,
  so the readout sees old and new data simultaneously.
  Avalanche uses reservoir sampling to keep the buffer representative.

  mem_size : total number of stored samples across all past tasks

No model architecture change is needed; replay acts purely at the data level.

Original repo: models/ESN/repo/experiments/splitmnist_esn.py  (strategy='replay')
Avalanche Replay: avalanche.training.Replay

Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
Input: MNIST pixels → sequence (B, image_size², 1).
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

from tasks.dataset_cl import MNIST_CL
from utils.metrics import CLMetrics, cohen_kappa
from models.ESN.esn_utils import build_model, predict_step
from shared.utils import collect_datasets
from clrnn.utils import get_strategy
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.plugins import EvaluationPlugin

_TT        = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
_NUM_TASKS = 5

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total   = 0
    for x, y in loader:
        batch_size   = y.size(0)
        num_correct  = num_correct + (predict_step(model, x) == y.numpy()).sum()
        num_total    = num_total   + batch_size
    return num_correct / num_total

def run_esn_replay_mnist(args, verbose = True, trial=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    device        = args.device
    data_root     = args.data_dir
    batch_size    = args.batch_size
    learning_rate = args.learning_rate
    image_size    = 28
    max_samples   = args.subset

    # ── Data ─────────────────────────────────────────────────────────────────
    mnist_train = MNIST_CL(data_root, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=2, image_size=image_size)
    mnist_test  = MNIST_CL(data_root, download=False, train=False,
                           output_size=2, image_size=image_size)


    train_datasets, val_datasets, test_datasets = collect_datasets(
        mnist_train, mnist_test, _TT, max_samples, batch_size, reshape=True)
    scenario = dataset_benchmark(train_datasets, test_datasets)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model     = build_model(input_size=image_size, args=args, device=device)
    opt       = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # ── CL strategy: Replay ───────────────────────────────────────────────────
    # mem_size controls the total replay buffer capacity across all past tasks.
    args_cl     = Namespace(**{**vars(args), "strategy": "replay"})
    cl_strategy = get_strategy(model, opt, criterion, EvaluationPlugin(), device, args_cl)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    # ── Task loop ─────────────────────────────────────────────────────────────
    for task_id, exp in enumerate(scenario.train_stream):
        task_name = f"{_TT[task_id][0]}/{_TT[task_id][1]}"
        if verbose:
            print(f"  Task {task_id+1}/{_NUM_TASKS}  [{task_name}]")

        # Pre-task accuracy (for FWT)
        loader_pt = DataLoader(test_datasets[task_id], batch_size=batch_size, shuffle=False)
        metrics.record_pretrain(task_id, calculate_accuracy(model, loader_pt))

        # Avalanche Replay: mixes current task data with buffered past samples each batch
        cl_strategy.train(exp)

        # Validation accuracy right after training this task
        loader_val = DataLoader(val_datasets[task_id], batch_size=batch_size, shuffle=False)
        subtask_val_accs.append(calculate_accuracy(model, loader_val))
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # R-matrix: evaluate on all tasks seen so far
        if verbose: print(f"  [test after task {task_id+1}]")
        for eval_id in range(task_id + 1):
            loader_te = DataLoader(test_datasets[eval_id], batch_size=batch_size, shuffle=False)
            all_predictions, all_labels = [], []
            for x, y in loader_te:
                all_predictions.extend(predict_step(model, x)); all_labels.extend(y.numpy())
            all_predictions = np.array(all_predictions);            num_of_correct_predictions = 0
            num_of_samples = len(all_labels)
            for i in range(num_of_samples):
                if all_predictions[i] == all_labels[i]:
                    num_of_correct_predictions += 1
            acc   = float(num_of_correct_predictions / num_of_samples)
            kappa = cohen_kappa(all_labels, all_predictions)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"    task {eval_id+1} [{_TT[eval_id][0]}/{_TT[eval_id][1]}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"  MNIST results  --  ESN-Replay")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
