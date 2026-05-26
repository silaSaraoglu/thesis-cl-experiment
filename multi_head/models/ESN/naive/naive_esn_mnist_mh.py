"""
ESN Naive Multi-Head -- Sequential MNIST  (TIL Setting 2).

Each task gets its own independent linear head on top of the shared frozen
reservoir.  Because the reservoir and all previous heads are never modified
after a task is learned, there is NO catastrophic forgetting.

At test time the oracle task ID is provided (Task-Incremental Learning upper
bound).  This model is identical in behaviour to the EWC / LwF / Replay MH
variants — CL regularisation is meaningless when heads are independent.

Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
Input: MNIST pixels flattened to a sequence (B, image_size², 1).
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

from tasks.dataset_cl import MNIST_CL
from utils.metrics import CLMetrics, cohen_kappa
from models.multi_head_utils import MultiHeadESN, _predict_mh
from shared.utils import collect_datasets

_TT        = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
_NUM_TASKS = 5

def run_esn_naive_mnist_mh(args, verbose = True, trial=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device     = args.device
    data_root  = args.data_dir
    image_size = 28
    max_samples = args.subset
    batch_size  = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    mnist_train = MNIST_CL(data_root, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=2, image_size=image_size)
    mnist_test  = MNIST_CL(data_root, download=False, train=False,
                           output_size=2, image_size=image_size)

    train_datasets, val_datasets, test_datasets = collect_datasets(
        mnist_train, mnist_test, _TT, max_samples, batch_size, reshape=True)

    model = MultiHeadESN(input_size=image_size, args=args, device=device)
    criterion = nn.CrossEntropyLoss()

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id in range(_NUM_TASKS):
        task_name = f"{_TT[task_id][0]}/{_TT[task_id][1]}"
        if verbose:
            print(f"  Task {task_id+1}/{_NUM_TASKS}  [{task_name}]")

        metrics.record_pretrain(task_id, 0.5)

        model.add_head(task_id)
        model.set_task(task_id)
        opt = torch.optim.Adam(model.heads[str(task_id)].parameters(), lr=learning_rate)

        loader_tr = DataLoader(train_datasets[task_id], batch_size=batch_size, shuffle=True, drop_last=False)
        for _ in range(epochs):
            model.train()
            for x, y in loader_tr:
                opt.zero_grad()
                criterion(model(x), y).backward()
                opt.step()

        loader_val = DataLoader(val_datasets[task_id], batch_size=batch_size, shuffle=False)
        preds, labels = _predict_mh(model, loader_val, device)
        num_of_correct_predictions = 0
        num_of_samples = len(labels)
        for i in range(num_of_samples):
            if preds[i] == labels[i]:
                num_of_correct_predictions += 1
        subtask_val_accs.append(float(num_of_correct_predictions / num_of_samples))
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if verbose: print(f"[test after task {task_id+1}]")
        for eval_id in range(task_id + 1):
            model.set_task(eval_id)
            loader_te = DataLoader(test_datasets[eval_id], batch_size=batch_size, shuffle=False)
            preds, labels = _predict_mh(model, loader_te, device)
            num_of_correct_predictions = 0
            num_of_samples = len(labels)
            for i in range(num_of_samples):
                if preds[i] == labels[i]:
                    num_of_correct_predictions += 1
            acc   = float(num_of_correct_predictions / num_of_samples)
            kappa = cohen_kappa(labels, preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"Task {eval_id+1} [{_TT[eval_id][0]}/{_TT[eval_id][1]}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"  MNIST results  --  ESN-Naive MH")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
