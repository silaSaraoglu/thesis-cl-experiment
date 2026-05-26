"""
ESN with SLDA -- Sequential MNIST.

CL strategy: Avalanche StreamingLDA via the original repo's get_strategy().
  Model: DeepReservoirClassifier wrapped with ESNWrapper(model, 'hidden').
    ESNWrapper hooks into the DeepReservoir layer and extracts states_last[-1]
    (last reservoir layer's final hidden state, shape B x esn_units) as the
    feature vector passed to StreamingLDA.
  StreamingLDA maintains ONE shared LDA (class means + covariance) updated
  incrementally across all tasks. No gradient training. No per-task classifier.
  No task ID used at test time — true single-head DIL setting.

Matches splitmnist_esn.py strategy='slda':
  model = ESNWrapper(DeepReservoirClassifier(...), 'hidden')
  cl_strategy = get_strategy(model, ..., args)  # args.strategy='slda'

Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
Input: MNIST pixels flattened to a sequence (B, image_size², 1).
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
from models.ESN.esn_utils import build_model
from shared.utils import collect_datasets
from clrnn.deep_esn import ESNWrapper
from clrnn.utils import get_strategy
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.plugins import EvaluationPlugin

_TT        = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
_NUM_TASKS = 5

def _predict(cl_strategy, loader):
    """Extract reservoir features then classify with the shared LDA."""
    all_predictions, all_labels = [], []
    device = next(cl_strategy.model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            feats = cl_strategy.model(x.to(device))
            preds = cl_strategy.predict(feats).argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_predictions), np.array(all_labels)

def run_esn_slda_mnist(args, verbose = True, trial=None):
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

    # ── Model: ESNWrapper for StreamingLDA feature extraction ────────────────
    # ESNWrapper hooks DeepReservoir.forward and captures states_last[-1]:
    # the last reservoir layer's final hidden state, shape (B, esn_units).
    reservoir = build_model(input_size=image_size, args=args, device=device)
    model     = ESNWrapper(reservoir, 'hidden')

    # ── Strategy: StreamingLDA via original repo's get_strategy ──────────────
    # StreamingLDA does not use gradient-based training; optimizer is a formality.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    args_cl   = Namespace(**{**vars(args),
        'strategy':    'slda',
        'input_size':  args.esn_units,
        'num_classes': 2,
        'shrinkage':   1e-4,
    })
    cl_strategy = get_strategy(model, optimizer, criterion,
                               EvaluationPlugin(), device, args_cl)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    # ── Task loop ─────────────────────────────────────────────────────────────
    for task_id, exp in enumerate(scenario.train_stream):
        task_name = f"{_TT[task_id][0]}/{_TT[task_id][1]}"
        if verbose:
            print(f"  Task {task_id+1}/{_NUM_TASKS}  [{task_name}]")

        # Pre-task: LDA not yet updated for this domain → 0.5
        metrics.record_pretrain(task_id, 0.5)

        # Incrementally update the shared LDA with this task's data
        cl_strategy.train(exp)

        # Validation accuracy with the updated shared LDA
        loader_val = DataLoader(val_datasets[task_id], batch_size=batch_size, shuffle=False)
        preds, labels = _predict(cl_strategy, loader_val)
        final_val_acc = float((preds == labels).mean())
        subtask_val_accs.append(final_val_acc)
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        if verbose:
            print(f"    StreamingLDA updated | val_acc={final_val_acc:.4f}")

        # R-matrix: same shared LDA applied to all tasks seen so far
        if verbose: print(f"  [test after task {task_id+1}]")
        for eval_id in range(task_id + 1):
            loader_te = DataLoader(test_datasets[eval_id], batch_size=batch_size, shuffle=False)
            preds, labels = _predict(cl_strategy, loader_te)
            acc   = float((preds == labels).mean())
            kappa = cohen_kappa(labels, preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"    task {eval_id+1} [{_TT[eval_id][0]}/{_TT[eval_id][1]}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"  MNIST results  --  ESN-SLDA")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
