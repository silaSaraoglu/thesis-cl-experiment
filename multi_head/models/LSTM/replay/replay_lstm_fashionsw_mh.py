"""
LSTM with Experience Replay -- MNIST  (TIL Setting 2).

CL strategy: Experience Replay.
  A fixed-size buffer stores mem_size samples from all past tasks combined.
  Each task slot gets mem_size // num_tasks_seen samples (balanced reservoir).
  New-task training uses the concatenation of current-task data + buffer,
  so the model sees old and new examples in every epoch.

Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
Input: MNIST pixels row-by-row (B, 28, 28) — 28 time steps × 28 pixel features.
"""
import os, sys
import torch



import torch.optim as optim
import optuna
import numpy as np
from shared.utils import MNIST_PERM
from torch.utils.data import DataLoader, TensorDataset

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import FashionMNISTSeason_CL
from shared.metrics import CLMetrics, cohen_kappa
from models.LSTM.multi_head_lstm_model import MultiHeadLSTM

_TT = [[0, 2], [3, 4], [5, 9]]
_NUM_TASKS = 3



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
            preds = model(x).argmax(1)
            for i in range(len(preds)):
                if preds[i] == y[i]:
                    num_correct += 1
            num_total += y.size(0)
    return num_correct / num_total


def run_lstm_replay_fashionsw_mh(args, verbose=True, trial=None):
    """LSTM + Replay on MNIST. Returns val_accs, test_accs, metrics, model, []."""
    hidden_size = args.hidden_size_rnn
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    max_samples = args.subset
    task_pairs = getattr(args, 'task_pairs', _TT)
    epochs = args.epochs
    max_grad_norm = 5.0
    image_size = 28
    output_size = 2

    device = args.device
    model = MultiHeadLSTM(image_size, hidden_size, device, batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    mnist_train = FashionMNISTSeason_CL(data_dir, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=output_size)
    mnist_test = FashionMNISTSeason_CL(data_dir, download=False, train=False,
                           output_size=output_size)
    mnist_train.set_holdout_config(
        holdout_n = getattr(args, 'holdout_n', 0),
        holdout_seed = getattr(args, 'holdout_seed', 0),
        use_holdout = getattr(args, 'use_holdout', False),
    )

    metrics = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    mem_size = 0  # set from first task's training data
    buf = []  # list of (X, Y) per past task; total ≤ mem_size via reservoir sampling

    for task_id in range(_NUM_TASKS):
        mnist_train.choose_subset(task_pairs[task_id])
        loader_train, loader_val = mnist_train.get_train_val_loader(max_samples=max_samples)
        if task_id == 0:
            mem_size = max(1, int(len(loader_train.dataset) * _NUM_TASKS * 0.1))

        metrics.record_pretrain(task_id, 0.5)  # no head yet in MH before add_head

        model.add_head(task_id)
        model.set_task(task_id)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Materialize current task's training data
        xs, ys = [], []
        for x, y in DataLoader(loader_train.dataset, batch_size=256, shuffle=False, drop_last=False):
            xs.append(x)
            ys.append(y)
        cur_X, cur_Y = torch.cat(xs), torch.cat(ys)

        # Build combined dataset: current task + replay buffer
        all_X = torch.cat([cur_X] + [b[0] for b in buf])
        all_Y = torch.cat([cur_Y] + [b[1] for b in buf])

        combined_loader = DataLoader(
            TensorDataset(all_X, all_Y), batch_size=batch_size, shuffle=True, drop_last=False)

        for _ in range(epochs):
            model.train()
            for x, y in combined_loader:
                x = _px(x).to(device)
                y = y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        model.set_task(task_id)
        subtask_val_accs.append(calculate_accuracy(model, loader_val))
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Store fixed per-task quota: mem_size split equally across all tasks
        perm = torch.randperm(cur_X.size(0))[:max(1, mem_size // _NUM_TASKS)]
        buf.append((cur_X[perm], cur_Y[perm]))

        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            model.set_task(eval_id)
            mnist_test.choose_subset(task_pairs[eval_id])
            loader_te = DataLoader(mnist_test, batch_size=batch_size,
                                   shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for x, y in loader_te:
                    x = _px(x).to(device)
                    preds = model(x).argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.numpy())
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            num_correct = 0
            for i in range(len(all_preds)):
                if all_preds[i] == all_labels[i]:
                    num_correct += 1
            acc = float(num_correct / len(all_labels))
            kappa = cohen_kappa(all_labels, all_preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            if verbose:
                print(f"    task {eval_id+1} acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [metrics.R[_NUM_TASKS-1].get(j, 0.0) for j in range(_NUM_TASKS)]
    if verbose:
        print()
        print("  MNIST results  --  LSTM-Replay")
        print(f"  Validation: {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test:       {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, model, []
