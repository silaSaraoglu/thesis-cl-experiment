"""
Joint Multi-Head LSTM -- MNIST  (TIL Setting 2, upper bound).

All task heads added upfront; shared body + all heads trained simultaneously
each epoch — oracle upper bound with no forgetting.
Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
"""
import os, sys, random
import torch



import torch.optim as optim
import numpy as np
from shared.utils import MNIST_PERM
from torch.utils.data import DataLoader, TensorDataset

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

def run_joint_lstm_fashion_mh(args, verbose = True):
    """Joint Multi-Head LSTM on MNIST. Returns val_accs, test_accs, metrics, model, []."""


    device = args.device
    batch_size = args.batch_size
    epochs = args.epochs
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

    # Add all heads and materialize data upfront before choose_subset changes the object.
    # Use a large batch loader (drop_last=False) so tiny subsets don't produce empty loaders.
    train_datasets, val_datasets = [], []
    for t in range(_NUM_TASKS):
        model.add_head(t)
        mnist_train.choose_subset(task_pairs[t])
        loader_train, loader_val = mnist_train.get_train_val_loader(max_samples=max_samples)
        for src, store in [(loader_train, train_datasets), (loader_val, val_datasets)]:
            tmp = DataLoader(src.dataset, batch_size=256, shuffle=False, drop_last=False)
            batch_inputs, batch_labels = [], []
            for x, y in tmp:
                batch_inputs.append(x)
                batch_labels.append(y)
            store.append(TensorDataset(torch.cat(batch_inputs), torch.cat(batch_labels)))

    loaders_train = [DataLoader(train_datasets[t], batch_size=batch_size, shuffle=True)
                     for t in range(_NUM_TASKS)]
    loaders_val = [DataLoader(val_datasets[t],   batch_size=batch_size, shuffle=False)
                     for t in range(_NUM_TASKS)]

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    metrics = CLMetrics(num_tasks=_NUM_TASKS, joint_mode=True)
    subtask_val_accs = []

    for _ in range(epochs):
        all_batches = [(t, batch) for t in range(_NUM_TASKS) for batch in loaders_train[t]]
        random.shuffle(all_batches)
        model.train()
        for task_id, (x, y) in all_batches:
            model.set_task(task_id)
            x = _px(x).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    for t in range(_NUM_TASKS):
        model.set_task(t)
        subtask_val_accs.append(calculate_accuracy(model, loaders_val[t]))

    if verbose: print(" [After joint training]")
    for eval_id in range(_NUM_TASKS):
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
        metrics.record(after_task=_NUM_TASKS - 1, eval_task=eval_id, acc=acc)
        metrics.record_kappa(after_task=_NUM_TASKS - 1, eval_task=eval_id, kappa=kappa)
        if verbose:
            print(f"    task {eval_id+1} acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [metrics.R[_NUM_TASKS-1].get(j, 0.0) for j in range(_NUM_TASKS)]
    if verbose:
        acc_final = 0.0
        avg_kappa = 0.0
        for j in range(_NUM_TASKS):
            acc_final += test_accs[j]
            avg_kappa += metrics.K[_NUM_TASKS-1].get(j, 0.0)
        acc_final = float(acc_final / _NUM_TASKS)
        avg_kappa = float(avg_kappa / _NUM_TASKS)
        print()
        print("  MNIST results  --  LSTM MH (Joint)  [oracle upper bound]")
        print("  NOTE: Only ACC_final and Cohen's kappa are valid for joint training.")
        print("  ACC_avg, BWT, FWT, Plasticity and Stability require sequential")
        print("  task evaluation (full R-matrix) and are not reported.")
        print(f"  ACC_final : {acc_final:.4f}")
        print(f"  Cohen's kappa : {avg_kappa:.4f}")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, model, []
