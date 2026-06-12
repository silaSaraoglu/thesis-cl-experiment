"""
ESN Naive Multi-Head -- Sequential MNIST  (TIL Setting 2).

Each task gets its own independent linear head on top of the shared frozen
reservoir.  Because the reservoir and all previous heads are never modified
after a task is learned, there is NO catastrophic forgetting.

At test time the oracle task ID is provided (Task-Incremental Learning upper
bound).  This model is identical in behaviour to the EWC / LwF / Replay MH
variants — CL regularisation is meaningless when heads are independent.

Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
Input: MNIST pixels row-by-row (B, 28, 28) — 28 time steps × 28 pixel features.
"""
import os, sys, warnings
import torch

import torch.nn as nn
import optuna
import numpy as np
from torch.utils.data import DataLoader

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import FashionMNISTSeason_CL
from shared.metrics import CLMetrics, cohen_kappa
from models.ESN.multi_head_esn_model import MultiHeadESN
from shared.utils import collect_datasets

_TT = [[0, 2], [3, 4], [5, 9]]
_NUM_TASKS = 3

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device)
            preds = model(x).argmax(1).cpu()
            for i in range(len(preds)):
                if preds[i] == y[i]:
                    num_correct += 1
            num_total += y.size(0)
    return num_correct / num_total

def run_esn_naive_fashionsw_mh(args, verbose = True, trial=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device = args.device
    data_root = args.data_dir
    image_size = 28
    max_samples = args.subset
    task_pairs = getattr(args, 'task_pairs', _TT)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    mnist_train = FashionMNISTSeason_CL(data_root, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=2)
    mnist_test = FashionMNISTSeason_CL(data_root, download=False, train=False,
                           output_size=2)
    mnist_train.set_holdout_config(
        holdout_n = getattr(args, "holdout_n", 0),
        holdout_seed = getattr(args, "holdout_seed", 0),
        use_holdout = getattr(args, "use_holdout", False),
    )

    train_datasets, val_datasets, test_datasets = collect_datasets(
        mnist_train, mnist_test, task_pairs, max_samples, batch_size, reshape=True)

    model = MultiHeadESN(input_size=image_size, args=args, device=device)
    criterion = nn.CrossEntropyLoss()

    metrics = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id in range(_NUM_TASKS):

        metrics.record_pretrain(task_id, 0.5)

        model.add_head(task_id)
        model.set_task(task_id)
        opt = torch.optim.Adam(model.heads[str(task_id)].parameters(), lr=learning_rate)

        loader_tr = DataLoader(train_datasets[task_id], batch_size=batch_size, shuffle=True, drop_last=False)
        for _ in range(epochs):
            model.train()
            for x, y in loader_tr:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                criterion(model(x), y).backward()
                opt.step()

        loader_val = DataLoader(val_datasets[task_id], batch_size=batch_size, shuffle=False)
        subtask_val_accs.append(calculate_accuracy(model, loader_val))
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            model.set_task(eval_id)
            loader_te = DataLoader(test_datasets[eval_id], batch_size=batch_size, shuffle=False)
            preds, labels = [], []
            model.eval()
            with torch.no_grad():
                for x, y in loader_te:
                    x = x.to(device)
                    preds.extend(model(x).argmax(1).cpu().numpy())
                    labels.extend(y.numpy())
            num_of_correct_predictions = 0
            num_of_samples = len(labels)
            for i in range(num_of_samples):
                if preds[i] == labels[i]:
                    num_of_correct_predictions += 1
            acc = float(num_of_correct_predictions / num_of_samples)
            kappa = cohen_kappa(labels, preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"Task {eval_id+1} [{task_pairs[eval_id][0]}/{task_pairs[eval_id][1]}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"  MNIST results  --  ESN-Naive MH")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
