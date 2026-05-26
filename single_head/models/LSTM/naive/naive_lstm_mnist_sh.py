"""
Naive Single-Head LSTM -- MNIST  (DIL Setting 1).

Single LSTM trained sequentially on all tasks with no CL strategy.
Used as the lower-bound comparison for GIM-LSTM.
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

from tasks.dataset_cl import MNIST_CL
from tasks.mnist.utils_single import accuracy
from utils.metrics import CLMetrics, cohen_kappa
from models.LSTM.model import build_lstm, lstm_forward

_TT        = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
_NUM_TASKS = 5

# calculation is based on batch size considering last batch size may differ than others
def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.squeeze(1)
            batch_size = y.size(0)
            num_correct =  num_correct + accuracy(lstm_forward(model, x), y) * batch_size
            num_total = num_total + batch_size
    return num_correct / num_total


def run_naive_lstm_mnist_sh(args, verbose=True, trial=None):
    """Naive Single-Head LSTM on MNIST. Returns val_accs, test_accs, metrics, model, []."""
    hidden_size   = args.hidden_size_rnn
    batch_size    = args.batch_size
    learning_rate = args.learning_rate
    data_dir      = args.data_dir
    max_samples   = args.subset
    image_size    = 28
    epochs        = args.epochs
    max_grad_norm = 5.0
    input_size    = image_size
    output_size   = 2

    model     = build_lstm(input_size, hidden_size, output_size, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    mnist_train = MNIST_CL(data_dir, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=output_size, image_size=image_size)
    mnist_test  = MNIST_CL(data_dir, download=False, train=False,
                           output_size=output_size, image_size=image_size)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id in range(_NUM_TASKS):
        mnist_train.choose_subset(_TT[task_id])
        loader_train, loader_val = mnist_train.get_train_val_loader(max_samples=max_samples)

        mnist_test.choose_subset(_TT[task_id])
        loader_pt = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=False)
        metrics.record_pretrain(task_id, calculate_accuracy(model, loader_pt))

        for _ in range(epochs):
            model.train()
            for x, y in loader_train:
                x = x.squeeze(1)
                optimizer.zero_grad()
                loss = criterion(lstm_forward(model, x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        subtask_val_accs.append(calculate_accuracy(model, loader_val)) # validation accuracy added after each task training for each training
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        for eval_id in range(task_id + 1):
            mnist_test.choose_subset(_TT[eval_id])
            loader_test = DataLoader(mnist_test, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for x, y in loader_test:
                    x = x.squeeze(1)
                    preds  = lstm_forward(model, x).argmax(dim=1).numpy()
                    labels = y.numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            all_preds  = np.array(all_preds)
            all_labels = np.array(all_labels)
            acc   = float((all_preds == all_labels).mean())
            kappa = cohen_kappa(all_labels, all_preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"    task {eval_id + 1} acc = {acc:.4f}  kappa = {kappa:.4f}")

    test_accs = [metrics.R[_NUM_TASKS - 1].get(j, 0.0) for j in range(_NUM_TASKS)]
    if verbose:
        print()
        print("  MNIST results  --  LSTM (Naive)")
        print(f"  Validation: {[round(v, 4) for v in subtask_val_accs]}")
        print(f"  Test:       {[round(t, 4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, model, []
