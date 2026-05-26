"""
GIM experiment runner -- WISDM (accelerometer).
3 binary-classification tasks split by activity pair:
  Task 1: Walking vs Jogging
  Task 2: Upstairs vs Downstairs
  Task 3: Sitting vs Standing
input_size=3 (x/y/z accelerometer channels), seq_len=128 timesteps.
"""
import os
import sys
import torch.nn.functional as F
import optuna
import numpy as np
from argparse import Namespace
from torch.utils.data import DataLoader

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from tasks.dataset_cl import WISDM_CL
from tasks.mnist.utils_single import train, test, accuracy, train_autoencoder, test_autoencoder
from experiment.CL_experiment import CL_Experiment
from utils.metrics import CLMetrics, cohen_kappa
from shared.utils import gim_predict

# WISDM label pairs per task (0=Walking,1=Jogging,2=Upstairs,3=Downstairs,4=Sitting,5=Standing)
_TT        = [[0, 1], [2, 3], [4, 5]]  # Walking/Jogging · Upstairs/Downstairs · Sitting/Standing
_NUM_TASKS = 3


def calculate_accuracy(train_models, variant, loader, device, output_size):
    num_correct = 0.0
    num_total   = 0
    for x, y in loader:
        batch_size   = y.size(0)
        _, acc = test(train_models, variant, x, y, accuracy, device, output_size)
        num_correct  = num_correct + acc * batch_size
        num_total    = num_total   + batch_size
    return num_correct / num_total

def run_gim_wisdm(variant, args, verbose = True, trial=None):
    """
    Train and evaluate one GIM run on WISDM.

    Returns
    -------
    val_accs           : list[float]
    test_accs          : list[float]
    metrics            : CLMetrics
    train_models
    train_autoencoders
    """
    hidden_size_rnn         = args.hidden_size_rnn
    hidden_sizes_lmn        = args.hidden_sizes_lmn
    hidden_size_autoencoder = args.hidden_size_autoencoder
    learning_rate           = args.learning_rate
    batch_size              = args.batch_size
    data_dir                = args.data_dir
    max_samples             = args.subset
    epochs                  = args.epochs
    max_grad_norm           = 5.0
    input_size              = 3
    output_size             = 2

    # ── Build exp_args matching the original repo's argparse Namespace ────────
    # CL_Experiment expects exactly these fields (see repo/experiment/CL_experiment.py).
    exp_args = Namespace(
        models=[variant],
        input_size=input_size,
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

    # CL_Experiment.create_models() builds the ALSTM/ALMN and one AE per task
    cl_exp = CL_Experiment(exp_args)
    device = cl_exp.get_device()
    train_models, train_autoencoders = cl_exp.create_models()

    # ── Data ─────────────────────────────────────────────────────────────────
    wisdm_train = WISDM_CL(data_dir, train=True,  download=False,
                         perc_val=0.25, batch_size=batch_size)
    wisdm_test  = WISDM_CL(data_dir, train=False, download=False)

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    # ── Task loop ─────────────────────────────────────────────────────────────
    for task_id in range(_NUM_TASKS):
        wisdm_train.choose_subset(_TT[task_id])
        loader_train, loader_val = wisdm_train.get_train_val_loader(max_samples=max_samples)

        task_name = WISDM_CL.TASK_NAMES.get(task_id + 1, str(_TT[task_id]))
        if verbose:
            print(f"")
            print(f"  Task {task_id + 1}/{_NUM_TASKS}  [{task_name}]")

        # Pre-task accuracy on this task before training it (for FWT)
        wisdm_test.choose_subset(_TT[task_id])
        loader_pt = DataLoader(wisdm_test, batch_size=batch_size,
                               shuffle=False, drop_last=False)
        metrics.record_pretrain(task_id, calculate_accuracy(train_models, variant, loader_pt, device, output_size))

        for _ in range(epochs):
            for x, y in loader_train:
                if x.size(0) != batch_size:
                    continue  # skip partial last batch (hidden state size mismatch)
                if train_autoencoders:
                    # Train the task-t AE to reconstruct task-t inputs (routing signal)
                    train_autoencoder(
                        train_autoencoders[0][task_id],
                        train_autoencoders[1][task_id],
                        x, F.mse_loss, device,
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
        if verbose:
            print(f"Test after task {task_id + 1}]")
        for eval_id in range(task_id + 1):
            wisdm_test.choose_subset(_TT[eval_id])
            loader_test = DataLoader(wisdm_test, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            for x, y in loader_test:
                if train_autoencoders:
                    # Route: pick the module whose AE best reconstructs this input
                    _, mod_id = test_autoencoder(train_autoencoders, x, F.mse_loss, device)
                else:
                    mod_id = None
                preds = gim_predict(train_models, x, task_id=mod_id)
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
            all_preds  = np.array(all_preds)
            all_labels = np.array(all_labels)
            acc   = float((all_preds == all_labels).mean())
            kappa = cohen_kappa(all_labels, all_preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            eval_name = WISDM_CL.TASK_NAMES.get(eval_id + 1, str(eval_id + 1))
            if verbose:
                print(f"Task {eval_id + 1} [{eval_name}]"
                      f" acc = {acc:.4f}  kappa = {kappa:.4f}")

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
        print(f"WISDM results  --  GIM-{variant.upper()}")
        print(f"Validation (right after each subtask):   {[round(v, 4) for v in subtask_val_accs]}")
        print(f"Test       (end of all subtasks):        {[round(t, 4) for t in test_accs]}")

    return subtask_val_accs, test_accs, metrics, train_models, train_autoencoders
