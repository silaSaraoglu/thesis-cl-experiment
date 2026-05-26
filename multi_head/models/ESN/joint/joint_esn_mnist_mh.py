"""
Joint Multi-Head ESN -- MNIST  (TIL Setting 2, upper bound).

All task heads added upfront and trained simultaneously each epoch.
Reservoir is frozen by design; only the per-task linear heads are updated.
Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
Input: MNIST pixels flattened to sequence (B, image_size², 1).
"""
import os, sys, warnings, random
import torch
import torch.nn as nn
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

def run_joint_esn_mnist_mh(args, verbose = True):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device      = args.device
    data_root   = args.data_dir
    image_size  = 28
    max_samples = args.subset
    batch_size  = args.batch_size
    epochs      = args.epochs

    mnist_train = MNIST_CL(data_root, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=2, image_size=image_size)
    mnist_test  = MNIST_CL(data_root, download=False, train=False,
                           output_size=2, image_size=image_size)

    train_datasets, val_datasets, test_datasets = collect_datasets(
        mnist_train, mnist_test, _TT, max_samples, batch_size, reshape=True)

    model = MultiHeadESN(input_size=image_size, args=args, device=device)
    criterion = nn.CrossEntropyLoss()

    for t in range(_NUM_TASKS):
        model.add_head(t)

    all_head_params = []
    for t in range(_NUM_TASKS):
        all_head_params.extend(list(model.heads[str(t)].parameters()))
    opt = torch.optim.Adam(all_head_params,
                           lr=args.learning_rate)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS, joint_mode=True)
    subtask_val_accs = []

    loaders_train = [DataLoader(train_datasets[t], batch_size=batch_size,
                                shuffle=True, drop_last=False)
                     for t in range(_NUM_TASKS)]

    for epoch in range(epochs):
        all_batches = [(t, batch) for t in range(_NUM_TASKS) for batch in loaders_train[t]]
        random.shuffle(all_batches)
        model.train()
        total_loss, n_batches = 0.0, 0
        for task_id, (x, y) in all_batches:
            model.set_task(task_id)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item(); n_batches += 1
        if verbose:
            print(f"    epoch {epoch}/{epochs} | loss={total_loss/n_batches if n_batches else 0:.4f}")

    for t in range(_NUM_TASKS):
        model.set_task(t)
        loader_val = DataLoader(val_datasets[t], batch_size=batch_size, shuffle=False)
        preds, labels = _predict_mh(model, loader_val, device)
        num_of_correct_predictions = 0
        num_of_samples = len(labels)
        for i in range(num_of_samples):
            if preds[i] == labels[i]:
                num_of_correct_predictions += 1
        subtask_val_accs.append(float(num_of_correct_predictions / num_of_samples))

    if verbose: print("  [test after joint training]")
    for eval_id in range(_NUM_TASKS):
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
        metrics.record(after_task=_NUM_TASKS - 1, eval_task=eval_id, acc=acc)
        metrics.record_kappa(after_task=_NUM_TASKS - 1, eval_task=eval_id, kappa=kappa)
        if verbose:
            print(f"    task {eval_id+1} [{_TT[eval_id][0]}/{_TT[eval_id][1]}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        acc_final = float(np.mean(test_accs))
        avg_kappa = float(np.mean([metrics.K[_NUM_TASKS-1].get(j, 0.0) for j in range(_NUM_TASKS)]))
        print()
        print("  MNIST results  --  ESN-Joint MH  [oracle upper bound]")
        print("  NOTE: Only ACC_final and Cohen's kappa are valid for joint training.")
        print("  ACC_avg, BWT, FWT, Plasticity and Stability require sequential")
        print("  task evaluation (full R-matrix) and are not reported.")
        print(f"  ACC_final : {acc_final:.4f}")
        print(f"  Cohen's kappa : {avg_kappa:.4f}")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
