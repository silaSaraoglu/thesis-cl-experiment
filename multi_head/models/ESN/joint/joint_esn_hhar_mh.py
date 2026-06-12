"""
Joint Multi-Head ESN -- HHAR  (TIL Setting 2, upper bound).

All device heads added upfront and trained simultaneously each epoch.
Reservoir frozen; only the per-device 6-way heads are updated.
Tasks = device models. Input: (B, 128, 3).
"""
import os, sys, warnings, random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import HHAR_CL
from shared.metrics import CLMetrics, cohen_kappa
from models.ESN.multi_head_esn_model import MultiHeadESN
from shared.utils import collect_datasets

_TASKS     = [0, 1, 2, 3]
_NUM_TASKS = 4

def calculate_accuracy(model, loader):
    num_correct = 0.0
    num_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device)
            preds = model(x).argmax(1).cpu()
            for i in range(len(preds)):
                if preds[i] == y[i]:
                    num_correct += 1
            num_total += y.size(0)
    return num_correct / num_total

def run_joint_esn_hhar_mh(args, verbose = True):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device      = args.device
    data_root   = args.data_dir
    max_samples = args.subset
    task_order  = getattr(args, 'task_pairs', _TASKS)
    batch_size  = args.batch_size
    epochs      = args.epochs

    hhar_train = HHAR_CL(data_root, train=True,  download=False,
                         perc_val=0.25, batch_size=batch_size)
    hhar_test  = HHAR_CL(data_root, train=False, download=False)
    hhar_train.set_holdout_config(
        holdout_n    = getattr(args, "holdout_n", 0),
        holdout_seed = getattr(args, "holdout_seed", 0),
        use_holdout  = getattr(args, "use_holdout", False),
    )

    train_datasets, val_datasets, test_datasets = collect_datasets(
        hhar_train, hhar_test, task_order, max_samples, batch_size)

    model = MultiHeadESN(input_size=3, args=args, device=device, num_classes=6)
    criterion = nn.CrossEntropyLoss()

    for t in range(_NUM_TASKS):
        model.add_head(t)

    all_head_params = []
    for t in range(_NUM_TASKS):
        all_head_params.extend(list(model.heads[str(t)].parameters()))
    opt = torch.optim.Adam(all_head_params, lr=args.learning_rate)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS, joint_mode=True)
    subtask_val_accs = []

    loaders_train = [DataLoader(train_datasets[t], batch_size=batch_size,
                                shuffle=True, drop_last=False)
                     for t in range(_NUM_TASKS)]

    for _ in range(epochs):
        all_batches = [(t, batch) for t in range(_NUM_TASKS) for batch in loaders_train[t]]
        random.shuffle(all_batches)
        model.train()
        for task_id, (x, y) in all_batches:
            model.set_task(task_id)
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()

    for t in range(_NUM_TASKS):
        model.set_task(t)
        loader_val = DataLoader(val_datasets[t], batch_size=batch_size, shuffle=False)
        subtask_val_accs.append(calculate_accuracy(model, loader_val))

    if verbose: print("  [After joint training]")
    for eval_id in range(_NUM_TASKS):
        model.set_task(eval_id)
        loader_te = DataLoader(test_datasets[eval_id], batch_size=batch_size, shuffle=False)
        preds, labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader_te:
                x = x.to(device)
                preds.extend(model(x).argmax(1).cpu().numpy())
                labels.extend(y.numpy())
        num_of_correct_predictions = 0
        num_of_samples = len(labels)
        for i in range(num_of_samples):
            if preds[i] == labels[i]:
                num_of_correct_predictions += 1
        acc   = float(num_of_correct_predictions / num_of_samples)
        kappa = cohen_kappa(labels, preds)
        metrics.record(after_task=_NUM_TASKS - 1, eval_task=eval_id, acc=acc)
        metrics.record_kappa(after_task=_NUM_TASKS - 1, eval_task=eval_id, kappa=kappa)
        eval_name = HHAR_CL.DEVICE_MODELS[task_order[eval_id]]
        if verbose:
            print(f"    task {eval_id+1} [{eval_name}] acc={acc:.4f}  kappa={kappa:.4f}")

    test_accs = [float(metrics.R[_NUM_TASKS-1].get(j, 0.0)) for j in range(_NUM_TASKS)]
    if verbose:
        acc_final = float(sum(test_accs) / _NUM_TASKS)
        avg_kappa = float(sum(metrics.K[_NUM_TASKS-1].get(j, 0.0) for j in range(_NUM_TASKS)) / _NUM_TASKS)
        print()
        print("  HHAR results  --  ESN-Joint MH  [oracle upper bound]")
        print(f"  ACC_final : {acc_final:.4f}")
        print(f"  Cohen's kappa : {avg_kappa:.4f}")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics
