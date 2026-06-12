"""
Naive Single-Head LSTM -- HHAR  (Domain-Incremental).

Single LSTM trained sequentially on all tasks with no CL strategy.
DIL: one shared 6-way head; tasks = device models (nexus4 -> s3 -> s3mini -> samsungold).
Labels (6 activities) are CONSTANT across tasks; only the device domain shifts.
Input: (B, 128, 3).
"""
import os
import sys
import torch
import torch.optim as optim
import optuna
import numpy as np
from torch.utils.data import DataLoader

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import HHAR_CL
from tasks.mnist.utils_single import accuracy
from shared.metrics import CLMetrics, cohen_kappa
from models.LSTM.model import SingleHeadLSTM

# HHAR tasks = device-model indices (0=nexus4, 1=s3, 2=s3mini, 3=samsungold)
_TASKS     = [0, 1, 2, 3]
_NUM_TASKS = 4

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total   = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            batch_size   = y.size(0)
            num_correct  = num_correct + accuracy(model(x), y) * batch_size
            num_total    = num_total + batch_size
    return num_correct / num_total

def run_naive_lstm_hhar_sh(args, verbose = True, trial=None):
    """Naive Single-Head LSTM on HHAR. Returns val_accs, test_accs, metrics, model, []."""
    hidden_size   = args.hidden_size_rnn
    batch_size    = args.batch_size
    learning_rate = args.learning_rate
    data_dir      = args.data_dir
    max_samples   = args.subset
    task_order    = getattr(args, 'task_pairs', _TASKS)
    epochs        = args.epochs
    max_grad_norm = 5.0
    input_size    = 3
    output_size   = 6

    model     = SingleHeadLSTM(input_size, hidden_size, output_size, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    hhar_train = HHAR_CL(data_dir, train=True,  download=False,
                         perc_val=0.25, batch_size=batch_size)
    hhar_test  = HHAR_CL(data_dir, train=False, download=False)
    hhar_train.set_holdout_config(
        holdout_n    = getattr(args, "holdout_n", 0),
        holdout_seed = getattr(args, "holdout_seed", 0),
        use_holdout  = getattr(args, "use_holdout", False),
    )

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id in range(_NUM_TASKS):
        hhar_train.choose_domain(task_order[task_id])
        loader_train, loader_val = hhar_train.get_train_val_loader(max_samples=max_samples)

        hhar_test.choose_domain(task_order[task_id])
        loader_pt = DataLoader(hhar_test, batch_size=batch_size, shuffle=False, drop_last=False)
        metrics.record_pretrain(task_id, calculate_accuracy(model, loader_pt))

        for _ in range(epochs):
            model.train()
            for x, y in loader_train:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        subtask_val_accs.append(calculate_accuracy(model, loader_val))
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            hhar_test.choose_domain(task_order[eval_id])
            loader_test = DataLoader(hhar_test, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for x, y in loader_test:
                    preds  = model(x).argmax(dim=1).numpy()
                    labels = y.numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            all_preds  = np.array(all_preds)
            all_labels = np.array(all_labels)
            acc = float((all_preds == all_labels).sum() / len(all_labels))
            kappa = cohen_kappa(all_labels, all_preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            eval_name = HHAR_CL.DEVICE_MODELS[task_order[eval_id]]
            if verbose:
                print(f"    task {eval_id + 1} [{eval_name}] acc = {acc:.4f}  kappa = {kappa:.4f}")

    test_accs = [metrics.R[_NUM_TASKS - 1].get(j, 0.0) for j in range(_NUM_TASKS)]
    if verbose:
        print(f"")
        print("  HHAR results  --  LSTM (Naive)")
        print(f"  Validation: {[round(v, 4) for v in subtask_val_accs]}")
        print(f"  Test:       {[round(t, 4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, model, []
