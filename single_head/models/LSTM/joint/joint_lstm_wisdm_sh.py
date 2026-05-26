"""
Joint Single-Head LSTM -- WISDM  (DIL Setting 1, upper bound).

Trains on all tasks simultaneously each epoch — oracle upper bound with no forgetting.
Tasks: 3 binary tasks — Walking/Jogging · Upstairs/Downstairs · Sitting/Standing.
Input: WISDM accelerometer signals, shape (B, 128, 3).
"""
import os
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from tasks.dataset_cl import WISDM_CL
from tasks.mnist.utils_single import accuracy
from utils.metrics import CLMetrics, cohen_kappa
from models.LSTM.model import build_lstm, lstm_forward

_TT        = [[0, 1], [2, 3], [4, 5]]
_NUM_TASKS = 3

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total   = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            batch_size   = y.size(0)
            num_correct  = num_correct + accuracy(lstm_forward(model, x), y) * batch_size
            num_total    = num_total + batch_size
    return num_correct / num_total

def run_joint_lstm_wisdm_sh(args, verbose = True):
    """Joint Single-Head LSTM on WISDM. Returns val_accs, test_accs, metrics, model, []."""
    hidden_size   = args.hidden_size_rnn
    batch_size    = args.batch_size
    learning_rate = args.learning_rate
    data_dir      = args.data_dir
    max_samples   = args.subset
    epochs        = args.epochs
    max_grad_norm = 5.0
    input_size    = 3
    output_size   = 2

    model     = build_lstm(input_size, hidden_size, output_size, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    wisdm_train = WISDM_CL(data_dir, train=True,  download=False,
                         perc_val=0.25, batch_size=batch_size)
    wisdm_test  = WISDM_CL(data_dir, train=False, download=False)

    train_datasets, val_datasets = [], []
    for t in range(_NUM_TASKS):
        wisdm_train.choose_subset(_TT[t])
        loader_train, loader_val = wisdm_train.get_train_val_loader(max_samples=max_samples)
        for src, store in [(loader_train, train_datasets), (loader_val, val_datasets)]:
            tmp = DataLoader(src.dataset, batch_size=256, shuffle=False, drop_last=False)
            batch_inputs, batch_labels = [], []
            for x, y in tmp:
                batch_inputs.append(x)
                batch_labels.append(y)
            store.append(TensorDataset(torch.cat(batch_inputs), torch.cat(batch_labels)))

    loader_joint = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size,
                              shuffle=True, drop_last=False)
    loaders_val  = [DataLoader(val_datasets[t], batch_size=batch_size, shuffle=False)
                    for t in range(_NUM_TASKS)]

    metrics          = CLMetrics(num_tasks=_NUM_TASKS, joint_mode=True)
    subtask_val_accs = []

    for _ in range(epochs):
        model.train()
        for x, y in loader_joint:
            optimizer.zero_grad()
            loss = criterion(lstm_forward(model, x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
    for t in range(_NUM_TASKS):
        subtask_val_accs.append(calculate_accuracy(model, loaders_val[t]))

    if verbose: print("  [test after joint training]")
    for eval_id in range(_NUM_TASKS):
        wisdm_test.choose_subset(_TT[eval_id])
        loader_test = DataLoader(wisdm_test, batch_size=batch_size,
                                 shuffle=False, drop_last=False)
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader_test:
                preds  = lstm_forward(model, x).argmax(dim=1).numpy()
                labels = y.numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc   = float((all_preds == all_labels).mean())
        kappa = cohen_kappa(all_labels, all_preds)
        metrics.record(after_task=_NUM_TASKS - 1, eval_task=eval_id, acc=acc)
        metrics.record_kappa(after_task=_NUM_TASKS - 1, eval_task=eval_id, kappa=kappa)
        eval_name = WISDM_CL.TASK_NAMES.get(eval_id + 1, str(eval_id + 1))
        if verbose:
            print(f"    task {eval_id + 1} [{eval_name}] acc = {acc:.4f}  kappa = {kappa:.4f}")

    test_accs = [metrics.R[_NUM_TASKS - 1].get(j, 0.0) for j in range(_NUM_TASKS)]
    if verbose:
        acc_final = float(np.mean(test_accs))
        avg_kappa = float(np.mean([metrics.K[_NUM_TASKS-1].get(j, 0.0) for j in range(_NUM_TASKS)]))
        print(f"")
        print("  WISDM results  --  LSTM (Joint)  [oracle upper bound]")
        print("  NOTE: Only ACC_final and Cohen's kappa are valid for joint training.")
        print("  ACC_avg, BWT, FWT, Plasticity and Stability require sequential")
        print("  task evaluation (full R-matrix) and are not reported.")
        print(f"  ACC_final : {acc_final:.4f}")
        print(f"  Cohen's kappa : {avg_kappa:.4f}")
        print(f"  Validation: {[round(v, 4) for v in subtask_val_accs]}")
        print(f"  Test:       {[round(t, 4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, model, []
