"""
GIM experiment runner -- Sequential MNIST.

GIM (Growing Input Modules) is an expandable architecture for continual learning.
Original paper: "Continual Learning with Gated Incremental Memories" (Cossu et al.)
Original repo:  models/GIM/repo/mnist.py

How GIM works:
  - ALSTM: each module is an LSTM
  ALMN: each module is an LMN (Linear Memory Network).
  - One autoencoder (AE) is trained alongside the main model for each task.
  - Module growing: if val_acc < threshold_acc after task t, a new module is added
    before moving to task t+1.  threshold_acc=1.01 (default) → always add one module
    per task (since accuracy ≤ 1.0 always falls below 1.01).
  - Test-time routing: test_autoencoder() runs each AE on the test input and picks
    the module whose AE has the lowest reconstruction error.
  - gim_predict() classifies using the selected module.

This file calls the original repo functions WITHOUT modification:
  train()            — one supervised gradient step on the RNN readout
  test()             — forward pass + accuracy
  train_autoencoder()— one reconstruction step on the task-t AE
  test_autoencoder() — picks the module with lowest reconstruction loss
  CL_Experiment      — creates models and autoencoders with the original factory

Tasks: 5 binary classification tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9).
Input: MNIST as row-based sequence (B, 28, 28) — 28 time steps × 28 pixel features.
"""
import os
import sys
import argparse
import numpy as np
import optuna

import torch


from torch.utils.data import DataLoader

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import FashionMNIST_CL
from tasks.mnist.utils_single import train, test, accuracy, train_autoencoder, test_autoencoder
from tasks.utils import MSEMasked
from experiment.CL_experiment import CL_Experiment
from shared.metrics import CLMetrics, cohen_kappa
from shared.utils import gim_predict, MNIST_PERM

_TT = [[0, 5], [1, 7], [2, 9], [6, 8]]
_NUM_TASKS = 4



def _px(x):
    # permuted MNIST from GIM paper, kept as (B,28,28)
    return x.reshape(x.size(0), -1)[:, MNIST_PERM].reshape(x.size(0), 28, 28)

def calculate_accuracy(train_models, variant, loader, device, output_size):
    num_correct = 0.0
    num_total = 0
    for x, y in loader:
        x = _px(x)
        batch_size = y.size(0)
        _, acc = test(train_models, variant, x, y, accuracy, device, output_size)
        num_correct = num_correct +  acc * batch_size
        num_total = num_total   + batch_size
    return num_correct / num_total

def run_gim_fashion(variant, args, verbose = True, trial=None):
    """
    Train and evaluate one GIM run using the original repo code directly.

    Parameters
    ----------
    variant : 'alstm' or 'almn'

    Returns
    -------
    val_accs           : list[float]  — final val acc after training each task
    test_accs          : list[float]  — test acc on each task after all tasks trained
    metrics            : CLMetrics    — full R matrix + kappa
    train_models       : dict         — original model dict (ALSTM or ALMN)
    train_autoencoders : list         — one AE per task
    """
    hidden_size_rnn = args.hidden_size_rnn
    hidden_sizes_lmn = [128]  # LMN/ALMN-only field; unused by the ALSTM variant we run
    hidden_size_autoencoder = args.hidden_size_autoencoder
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    data_dir = args.data_dir
    max_samples = args.subset
    task_pairs = getattr(args, 'task_pairs', _TT)
    epochs = args.epochs
    max_grad_norm = 5.0
    image_size = 28
    input_size = image_size
    output_size = 2

    exp_args = argparse.Namespace(
        models=[variant],
        input_size = image_size,
        output_size=output_size,
        hidden_size_rnn=hidden_size_rnn,
        hidden_sizes_lmn=hidden_sizes_lmn,
        memory_size_lmn=128,
        hidden_size_autoencoder=hidden_size_autoencoder,
        type_A=False,
        feed_mem=False,
        orthogonal=False,
        orthogonal_loss=False,
        learning_rate=learning_rate,
        weight_decay=0.0,
        batch_size=batch_size,
        lr_ae=1e-4,
        decay_ae=0.0,
        tasks=list(range(1, _NUM_TASKS + 1)),
        load=False,
        plot_folder="plots/tmp/",
    )

    cl_exp = CL_Experiment(exp_args)
    device = cl_exp.get_device()
    train_models, train_autoencoders = cl_exp.create_models()

    mnist_train = FashionMNIST_CL(data_dir, download=False, train=True,
                           perc_val=0.25, batch_size=batch_size, output_size=output_size)
    mnist_test = FashionMNIST_CL(data_dir, download=False, train=False, output_size=output_size)
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


        # Pre-task accuracy on this task before training it (for FWT)
        mnist_test.choose_subset(task_pairs[task_id])
        loader_pt = DataLoader(mnist_test, batch_size=batch_size,
                               shuffle=False, drop_last=False)
        metrics.record_pretrain(task_id, calculate_accuracy(train_models, variant, loader_pt, device, output_size))

        for _ in range(epochs):
            for x, y in loader_train:
                x = _px(x)
                if train_autoencoders:
                    train_autoencoder(
                        train_autoencoders[0][task_id],
                        train_autoencoders[1][task_id],
                        x, MSEMasked, device,
                    )
                train(train_models, variant, x, y, accuracy, device, output_size,
                      max_grad_norm)
        final_val_acc = calculate_accuracy(train_models, variant, loader_val, device, output_size)
        subtask_val_accs.append(final_val_acc)
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # R-matrix: evaluate on all tasks seen so far
        # test_autoencoder selects the module with the lowest AE reconstruction error
        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            mnist_test.choose_subset(task_pairs[eval_id])
            loader_test = DataLoader(mnist_test, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            for x, y in loader_test:
                x = _px(x)
                if train_autoencoders:
                    # Route: pick the module whose AE best reconstructs this input
                    _, mod_id = test_autoencoder(train_autoencoders, x, MSEMasked, device)
                else:
                    mod_id = None
                preds = gim_predict(train_models, x, task_id=mod_id)
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
                print(f"Task {eval_id + 1} acc = {acc:.4f}  kappa = {kappa:.4f}")

        # Module growing: add a new module if val_acc < threshold_acc
        # threshold_acc=1.01 by default → always adds one module per task
        if task_id < _NUM_TASKS - 1 and final_val_acc < 1.01:
            if variant == "almn":
                n = len(train_models["almn"][0].lmns)
                if verbose:
                    print(f"  [GIM-ALMN] Added module {n + 1} for task {task_id + 1} "
                          f"(val_acc {final_val_acc:.3f} < 1.01)")
                train_models["almn"][0].add_new_module(train_models["almn"][1])
            else:
                n = len(train_models["alstm"][0].lstms)
                if verbose:
                    print(f"  [GIM-ALSTM] Added module {n + 1} for task {task_id + 1} "
                          f"(val_acc {final_val_acc:.3f} < 1.01)")
                train_models["alstm"][0].add_new_module(train_models["alstm"][1])

    test_accs = [metrics.R[_NUM_TASKS - 1].get(j, 0.0) for j in range(_NUM_TASKS)]

    if verbose:
        print(f"")
        print(f"Table I results  --  GIM-{variant.upper()}")
        print(f"Validation (right after each subtask):   {[round(v, 4) for v in subtask_val_accs]}")
        print(f"Test       (end of all subtasks):        {[round(t, 4) for t in test_accs]}")

    return subtask_val_accs, test_accs, metrics, train_models, train_autoencoders
