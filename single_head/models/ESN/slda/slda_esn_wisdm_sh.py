"""
ESN with SLDA -- WISDM.

CL strategy: Avalanche StreamingLDA via the original repo's get_strategy().
  Model: DeepReservoirClassifier wrapped with ESNWrapper(model, 'hidden').
  StreamingLDA maintains ONE shared LDA updated incrementally across all tasks.
  No task ID used at test time — true single-head DIL setting.

Tasks: 3 binary tasks — Walking/WalkingUp · WalkingDown/Sitting · Standing/Laying.
Input: WISDM accelerometer signals (B, 128, 3); features: (B, esn_units).
See esn_experiment_mnist.py for full strategy documentation.
Original repo: models/ESN/repo/experiments/splitmnist_esn.py  (strategy='slda')
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

from tasks.dataset_cl import WISDM_CL
from utils.metrics import CLMetrics, cohen_kappa
from models.ESN.esn_utils import build_model
from shared.utils import collect_datasets
from clrnn.deep_esn import ESNWrapper
from clrnn.utils import get_strategy
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.plugins import EvaluationPlugin

# WISDM label pairs per task (0=Walking,1=Jogging,2=Upstairs,3=Downstairs,4=Sitting,5=Standing)
_TT        = [[0, 1], [2, 3], [4, 5]]  # Walking/Jogging · Upstairs/Downstairs · Sitting/Standing
_NUM_TASKS = 3

def _predict(cl_strategy, loader):
    all_predictions, all_labels = [], []
    device = next(cl_strategy.model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            feats = cl_strategy.model(x.to(device))
            preds = cl_strategy.predict(feats).argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_predictions), np.array(all_labels)

def run_esn_slda_wisdm(args, verbose = True, trial=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    device        = args.device
    data_root     = args.data_dir
    batch_size    = args.batch_size
    learning_rate = args.learning_rate
    max_samples   = args.subset

    # ── Data ─────────────────────────────────────────────────────────────────
    wisdm_train = WISDM_CL(data_root, train=True,  download=False,
                         perc_val=0.25, batch_size=batch_size)
    wisdm_test  = WISDM_CL(data_root, train=False, download=False)

    train_datasets, val_datasets, test_datasets = collect_datasets(
        wisdm_train, wisdm_test, _TT, max_samples, batch_size)
    scenario = dataset_benchmark(train_datasets, test_datasets)

    # ── Model: ESNWrapper for StreamingLDA feature extraction ────────────────
    reservoir = build_model(input_size=3, args=args, device=device)
    model     = ESNWrapper(reservoir, 'hidden')

    # ── Strategy: StreamingLDA via original repo's get_strategy ──────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    args_cl   = Namespace(**{**vars(args),
        'strategy':    'slda',
        'input_size':  args.esn_units,
        'num_classes': 2,
        'shrinkage':   1e-5,
    })
    cl_strategy = get_strategy(model, optimizer, criterion,
                               EvaluationPlugin(), device, args_cl)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    # ── Task loop ─────────────────────────────────────────────────────────────
    for task_id, exp in enumerate(scenario.train_stream):
        task_name = WISDM_CL.TASK_NAMES.get(task_id + 1, str(_TT[task_id]))
        if verbose:
            print(f"  Task {task_id+1}/{_NUM_TASKS}  [{task_name}]")

        metrics.record_pretrain(task_id, 0.5)

        cl_strategy.train(exp)

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

        if verbose: print(f"  [test after task {task_id+1}]")
        for eval_id in range(task_id + 1):
            loader_te = DataLoader(test_datasets[eval_id], batch_size=batch_size, shuffle=False)
            preds, labels = _predict(cl_strategy, loader_te)
            acc   = float((preds == labels).mean())
            kappa = cohen_kappa(labels, preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            eval_name = WISDM_CL.TASK_NAMES.get(eval_id + 1, str(eval_id + 1))
            if verbose:
                print(f"    task {eval_id+1} [{eval_name}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"  WISDM results  --  ESN-SLDA")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
