"""
LSTM with Experience Replay -- Fashion-MNIST (summer-vs-winter)  (TIL Setting 2).

CL strategy: Experience Replay.
  A fixed-size (mem_size) balanced reservoir buffer, mirroring the ESN-Replay
  path's Avalanche memory (ExperienceBalancedBuffer + reservoir sampling): the
  buffer stays full and is split evenly across the tasks seen so far
  (mem_size // k each after k tasks), so after task 1 it holds mem_size task-1
  samples and each later task trims the earlier ones to make room. New-task
  training uses the concatenation of current-task data + buffer, so the model
  sees old and new examples in every epoch.

Tasks: 3 binary summer-vs-winter tasks within a garment category
(Tshirt/Pullover, Dress/Coat, Sandal/Ankle-boot).
Input: Fashion-MNIST pixels row-by-row (B, 28, 28) — 28 time steps × 28 pixel features.
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
import shared.utils

from shared.dataset_cl import FashionMNISTSeason_CL
from shared.metrics import CLMetrics, cohen_kappa
from shared.replay_buffer import BalancedReservoirBuffer
from models.LSTM.multi_head_lstm_model import MultiHeadLSTM

_TT = [[0, 2], [3, 4], [5, 9]]
_NUM_TASKS = 3



def px(x):
    # permuted MNIST from GIM paper, kept as (B,28,28)
    return x.reshape(x.size(0), -1)[:, MNIST_PERM].reshape(x.size(0), 28, 28)

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = px(x).to(model.device)
            y = y.to(model.device)
            preds = model(x).argmax(1)
            for i in range(len(preds)):
                if preds[i] == y[i]:
                    num_correct += 1
            num_total += y.size(0)
    return num_correct / num_total


def run_lstm_replay_fashionsw_mh(args, verbose=True, trial=None):
    """LSTM + Replay on Fashion-SW. Returns val_accs, test_accs, metrics, model, []."""
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

    # Size the replay buffer as the sum of 10% of each task's training split. The
    # pre-pass only reads split *sizes*; get_train_val_loader draws its split from
    # numpy's global RNG, so we snapshot/restore that state to leave the main loop's
    # splits unchanged.
    _rng_state = np.random.get_state()
    mem_size = 0
    for _tp in task_pairs:
        mnist_train.choose_subset(_tp)
        _lt, _ = mnist_train.get_train_val_loader(max_samples=max_samples)
        mem_size += int(len(_lt.batch_sampler.sampler) * 0.1)
    mem_size = max(1, mem_size)
    np.random.set_state(_rng_state)
    replay_buffer = BalancedReservoirBuffer(mem_size)  # mirrors the ESN-Replay (Avalanche) memory

    for task_id in range(_NUM_TASKS):
        mnist_train.choose_subset(task_pairs[task_id])
        loader_train, loader_val = mnist_train.get_train_val_loader(max_samples=max_samples)

        metrics.record_pretrain(task_id, 0.5)  # no head yet in MH before add_head

        model.add_head(task_id)
        model.set_task(task_id)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Materialize current task's training data
        xs, ys = [], []
        for x, y in loader_train:   # train split only (respects sampler + max_samples subset)
            xs.append(x)
            ys.append(y)
        cur_X, cur_Y = torch.cat(xs), torch.cat(ys)

        # Combined dataset: current task (head=task_id) + replay buffer, with each
        # sample tagged by the head of the task it came from, so multi-head replay
        # routes every replayed sample through its own head (not the current one).
        cur_T = torch.full((cur_X.size(0),), task_id, dtype=torch.long)
        buf_X, buf_Y, buf_T = replay_buffer.data_with_tasks()
        if buf_X is not None:
            all_X = torch.cat([cur_X, buf_X])
            all_Y = torch.cat([cur_Y, buf_Y])
            all_T = torch.cat([cur_T, buf_T])
        else:
            all_X, all_Y, all_T = cur_X, cur_Y, cur_T

        combined_loader = DataLoader(
            TensorDataset(all_X, all_Y, all_T), batch_size=batch_size, shuffle=True, drop_last=False)

        for _ in range(epochs):
            model.train()
            for x, y, t in combined_loader:
                x = px(x).to(device)
                y = y.to(device)
                optimizer.zero_grad()
                # Route each task's samples through its own head; weight per-head losses
                # by sample count / batch size to match one mean-reduced CrossEntropy
                # over the whole mini-batch.
                loss = 0.0
                for tid in torch.unique(t):
                    m = (t == tid)
                    model.set_task(int(tid))
                    loss = loss + criterion(model(x[m]), y[m]) * int(m.sum())
                loss = loss / x.size(0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        model.set_task(task_id)
        subtask_val_accs.append(calculate_accuracy(model, loader_val))
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Reservoir update + rebalance (mirrors Avalanche ExperienceBalancedBuffer)
        replay_buffer.update(cur_X, cur_Y, task_id)

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
                    x = px(x).to(device)
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
        print("  Fashion-SW results  --  LSTM-Replay")
        print(f"  Validation: {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test:       {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, model, []
