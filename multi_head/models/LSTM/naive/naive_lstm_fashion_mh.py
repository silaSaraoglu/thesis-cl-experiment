"""
Naive Multi-Head LSTM -- MNIST  (TIL Setting 2).

Shared LSTM body + per-task linear head (ModuleDict).
Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
"""
import os, sys
import torch

import torch.optim as optim
import optuna
import numpy as np
from shared.utils import MNIST_PERM
from torch.utils.data import DataLoader

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import FashionMNIST_CL
from tasks.mnist.utils_single import accuracy
from shared.metrics import CLMetrics, cohen_kappa
from models.LSTM.multi_head_lstm_model import MultiHeadLSTM

_TT = [[0, 5], [1, 7], [2, 9], [6, 8]]
_NUM_TASKS = 4


def _px(x):
    # permuted MNIST from GIM paper, kept as (B,28,28)
    return x.reshape(x.size(0), -1)[:, MNIST_PERM].reshape(x.size(0), 28, 28)

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = _px(x).to(model.device)
            y = y.to(model.device)
            batch_size = y.size(0)
            num_correct = num_correct + accuracy(model(x), y) * batch_size
            num_total = num_total + batch_size
    return num_correct / num_total

def run_naive_lstm_fashion_mh(args, verbose = True, trial=None):
    """Naive Multi-Head LSTM on MNIST. Returns val_accs, test_accs, metrics, model, []."""

    device = args.device
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    image_size = 28
    model = MultiHeadLSTM(image_size, args.hidden_size_rnn, device, batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    data_root = args.data_dir
    max_samples = args.subset
    task_pairs = getattr(args, 'task_pairs', _TT)

    mnist_train = FashionMNIST_CL(data_root, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=2)
    mnist_test = FashionMNIST_CL(data_root, download=False, train=False,
                           output_size=2)
    mnist_train.set_holdout_config(
        holdout_n = getattr(args, "holdout_n", 0),
        holdout_seed = getattr(args, "holdout_seed", 0),
        use_holdout = getattr(args, "use_holdout", False),
    )

    metrics = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id in range(_NUM_TASKS):
        mnist_train.choose_subset(task_pairs[task_id])
        loader_train, loader_val = mnist_train.get_train_val_loader(max_samples=max_samples)

        metrics.record_pretrain(task_id, 0.5)
        model.add_head(task_id)
        model.set_task(task_id)

        body_params = list(model.rnn_module.parameters())
        head_params = list(model.heads[str(task_id)].parameters())
        optimizer = optim.Adam(body_params + head_params, lr=learning_rate)

        for _ in range(epochs):
            model.train()
            for x, y in loader_train:
                x = _px(x).to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        final_val_acc = calculate_accuracy(model, loader_val)
        subtask_val_accs.append(final_val_acc)
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            model.set_task(eval_id)
            mnist_test.choose_subset(task_pairs[eval_id])
            loader_test = DataLoader(mnist_test, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for x, y in loader_test:
                    x = _px(x).to(device)
                    preds = model(x).argmax(dim=1).cpu().numpy()
                    labels = y.numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            num_of_correct_predictions = 0
            num_of_samples = len(all_labels)
            for i in range(num_of_samples):
                if all_preds[i] == all_labels[i]:
                    num_of_correct_predictions += 1
            acc = float(num_of_correct_predictions / num_of_samples)
            kappa = cohen_kappa(all_labels, all_preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"Task {eval_id+1} acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [metrics.R[_NUM_TASKS-1].get(j, 0.0) for j in range(_NUM_TASKS)]
    if verbose:
        print()
        print("MNIST results  --  LSTM MH (Naive)")
        print(f"Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, model, []
