"""
LSTM with Experience Replay -- Fashion-MNIST Summer-vs-Winter  (DIL Setting 1).

CL strategy: Experience Replay.
  A fixed-size (mem_size) balanced reservoir buffer, mirroring the ESN-Replay
  path's Avalanche memory (ExperienceBalancedBuffer + reservoir sampling): the
  buffer stays full and is split evenly across the tasks seen so far
  (mem_size // k each after k tasks), so after task 1 it holds mem_size task-1
  samples and each later task trims the earlier ones to make room. New-task
  training mixes a fixed amount of new and replayed samples in every mini-batch
  (batch_size new + batch_size memory, memory oversampled when smaller),
  mirroring Avalanche's ReplayDataLoader, so the model sees old and new examples
  in every step.

Tasks: 3 binary summer-vs-winter tasks (e.g. Tshirt/Pullover, Dress/Coat,
  Sandal/Ankle-boot).
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
from tasks.mnist.utils_single import accuracy
from shared.metrics import CLMetrics, cohen_kappa
from shared.replay_buffer import BalancedReservoirBuffer
from models.LSTM.model import SingleHeadLSTM

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
            x = px(x)
            batch_size = y.size(0)
            num_correct = num_correct + accuracy(model(x), y) * batch_size
            num_total = num_total   + batch_size
    return num_correct / num_total


def run_lstm_replay_fashionsw_sh(args, verbose=True, trial=None):
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

    model = SingleHeadLSTM(image_size, hidden_size, output_size, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    mnist_train = FashionMNISTSeason_CL(data_dir, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size,
                           output_size=output_size)
    mnist_test = FashionMNISTSeason_CL(data_dir, download=False, train=False,
                           output_size=output_size)
    
    # splitting data into train and validation sets so there is no data leakage 
    mnist_train.set_holdout_config(
        holdout_n = getattr(args, 'holdout_n', 0),
        holdout_seed = getattr(args, 'holdout_seed', 0),
        use_holdout = getattr(args, 'use_holdout', False),
    )

    metrics = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    # Size the replay buffer like the ESN-Replay path: the sum of 10% of each
    # task's training split, not task-0 x num_tasks (which assumes all tasks are
    # the same size). The pre-pass only reads split *sizes*; get_train_val_loader
    # draws its split from numpy's global RNG, so we snapshot/restore that state
    # to leave the actual splits in the main loop unchanged.
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

        # Pre-task accuracy (for FWT)
        mnist_test.choose_subset(task_pairs[task_id])
        loader_pt = DataLoader(mnist_test, batch_size=batch_size,
                               shuffle=False, drop_last=False)
        metrics.record_pretrain(task_id, calculate_accuracy(model, loader_pt))

        # Materialize current task's training data
        xs, ys = [], []
        for x, y in loader_train:   # train split only (respects sampler + max_samples subset)
            xs.append(x)
            ys.append(y)
        cur_X, cur_Y = torch.cat(xs), torch.cat(ys)

        # Separate current + memory loaders so each mini-batch mixes a fixed
        # amount of new and replayed samples (mirrors Avalanche's
        # ReplayDataLoader: batch_size new + batch_size_mem memory per batch,
        # with batch_size_mem == batch_size). The memory loader is re-cycled
        # whenever it runs dry, reproducing oversample_small_tasks=True.
        cur_loader = DataLoader(
            TensorDataset(cur_X, cur_Y), batch_size=batch_size, shuffle=True, drop_last=False)
        buf_X, buf_Y = replay_buffer.data()
        mem_loader = None
        if buf_X is not None:
            mem_loader = DataLoader(
                TensorDataset(buf_X, buf_Y), batch_size=batch_size, shuffle=True, drop_last=False)

        for _ in range(epochs):
            model.train()
            mem_iter = iter(mem_loader) if mem_loader is not None else None
            for x, y in cur_loader:
                if mem_iter is not None:
                    try:
                        mx, my = next(mem_iter)
                    except StopIteration:
                        mem_iter = iter(mem_loader)
                        mx, my = next(mem_iter)
                    x = torch.cat([x, mx])
                    y = torch.cat([y, my])
                x = px(x)
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

        # Reservoir update + rebalance (mirrors Avalanche ExperienceBalancedBuffer)
        replay_buffer.update(cur_X, cur_Y)

        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            mnist_test.choose_subset(task_pairs[eval_id])
            loader_te = DataLoader(mnist_test, batch_size=batch_size,
                                   shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for x, y in loader_te:
                    x = px(x)
                    preds = model(x).argmax(dim=1).numpy()
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
