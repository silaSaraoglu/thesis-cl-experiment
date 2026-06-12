"""
GIM Multi-Head experiment runner -- WISDM  (TIL Setting 2).

GIM architecture is unchanged: modules grow task-by-task.
Test-time routing uses oracle task ID instead of AE reconstruction error.

Tasks: 3 binary tasks — Walking/Jogging · Upstairs/Downstairs · Sitting/Standing.
Input: WISDM accelerometer signals, shape (B, 128, 3).
"""
import os, sys
import optuna
import numpy as np
from argparse import Namespace
from torch.utils.data import DataLoader

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from shared.dataset_cl import WISDM_CL
from tasks.mnist.utils_single import train, test, accuracy
from experiment.CL_experiment import CL_Experiment
from shared.metrics import CLMetrics, cohen_kappa
from shared.utils import gim_predict

# WISDM label pairs per task (0=Walking,1=Jogging,2=Upstairs,3=Downstairs,4=Sitting,5=Standing)
_TT        = [[0, 1], [2, 3], [4, 5]]  # Walking/Jogging · Upstairs/Downstairs · Sitting/Standing
_NUM_TASKS = 3

def calculate_accuracy(train_models, variant, loader, device, output_size, module_id):
    num_correct, num_total = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        _, acc = test(train_models, variant, x, y, accuracy, device, output_size, module_id=module_id)
        num_correct += acc * y.size(0)
        num_total   += y.size(0)
    return num_correct / num_total

def run_gim_wisdm_mh(variant, args, verbose = True, trial=None):
    """
    Train and evaluate one GIM run on WISDM using oracle task routing (TIL).

    Returns
    -------
    val_accs, test_accs, metrics, train_models, train_autoencoders
    """


    data_root   = args.data_dir
    max_samples = args.subset
    task_pairs = getattr(args, 'task_pairs', _TT)
    batch_size  = args.batch_size
    epochs      = args.epochs
    max_grad_norm = 5.0

    exp_args = Namespace(
        models=[variant],
        input_size=3,
        output_size=2,
        hidden_size_rnn=args.hidden_size_rnn,
        hidden_sizes_lmn=[128],  # LMN/ALMN-only field; unused by the ALSTM variant we run
        memory_size_lmn=128,
        hidden_size_autoencoder=args.hidden_size_autoencoder,
        type_A=False,
        feed_mem=False,
        orthogonal=False,
        orthogonal_loss=False,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        batch_size=batch_size,
        lr_ae=1e-4,
        decay_ae=0.0,
        tasks=list(range(1, _NUM_TASKS + 1)),
        cuda=args.device.type == 'cuda',
        load=False,
        plot_folder="plots/tmp/",
    )

    cl_exp = CL_Experiment(exp_args)
    device = cl_exp.get_device()
    train_models, train_autoencoders = cl_exp.create_models()

    wisdm_train = WISDM_CL(data_root, train=True,  download=False,
                         perc_val=0.25, batch_size=batch_size)
    wisdm_test  = WISDM_CL(data_root, train=False, download=False)
    wisdm_train.set_holdout_config(
        holdout_n    = getattr(args, "holdout_n", 0),
        holdout_seed = getattr(args, "holdout_seed", 0),
        use_holdout  = getattr(args, "use_holdout", False),
    )

    metrics          = CLMetrics(num_tasks=_NUM_TASKS)
    subtask_val_accs = []

    for task_id in range(_NUM_TASKS):
        wisdm_train.choose_subset(task_pairs[task_id])
        loader_train, loader_val = wisdm_train.get_train_val_loader(max_samples=max_samples)


        wisdm_test.choose_subset(task_pairs[task_id])
        loader_pt = DataLoader(wisdm_test, batch_size=batch_size,
                               shuffle=False, drop_last=False)
        num_correct, num_total = 0.0, 0
        for x, y in loader_pt:
            x = x.to(device)
            _, acc = test(train_models, variant, x, y, accuracy, device, 2, module_id=task_id)
            num_correct += acc * y.size(0)
            num_total += y.size(0)
        metrics.record_pretrain(task_id, num_correct / num_total)

        for _ in range(epochs):
            for x, y in loader_train:
                if x.size(0) != batch_size:
                    continue
                x = x.to(device)
                train(train_models, variant, x, y, accuracy, device, 2,
                      max_grad_norm)

        final_val_acc = calculate_accuracy(train_models, variant, loader_val, device, 2, task_id)
        subtask_val_accs.append(final_val_acc)
        if trial is not None:
            trial.report(float(np.mean(subtask_val_accs)), task_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if verbose: print(f"  [After task {task_id+1}/{_NUM_TASKS}]")
        for eval_id in range(task_id + 1):
            wisdm_test.choose_subset(task_pairs[eval_id])
            loader_test = DataLoader(wisdm_test, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
            all_preds, all_labels = [], []
            for x, y in loader_test:
                x = x.to(device)
                # MH: oracle routing — pass eval_id directly as module_id
                preds = gim_predict(train_models, x, task_id=eval_id)
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
            num_of_correct_predictions = 0
            num_of_samples = len(all_labels)
            for i in range(num_of_samples):
                if all_preds[i] == all_labels[i]:
                    num_of_correct_predictions += 1
            acc   = float(num_of_correct_predictions / num_of_samples)
            kappa = cohen_kappa(all_labels, all_preds)
            metrics.record(after_task=task_id, eval_task=eval_id, acc=acc)
            metrics.record_kappa(after_task=task_id, eval_task=eval_id, kappa=kappa)
            eval_name = WISDM_CL.TASK_NAMES.get(eval_id + 1, str(eval_id + 1))
            if verbose:
                print(f"    task {eval_id+1} [{eval_name}] acc={acc:.4f}  kappa={kappa:.4f}")

        if task_id < _NUM_TASKS - 1 and final_val_acc < 1.01:
            if variant == "almn":
                n = len(train_models["almn"][0].lmns)
                if verbose:
                    print(f"  [GIM-ALMN MH] Added module {n+1} for task {task_id+1}")
                train_models["almn"][0].add_new_module(train_models["almn"][1])
            else:
                n = len(train_models["alstm"][0].lstms)
                if verbose:
                    print(f"  [GIM-ALSTM MH] Added module {n+1} for task {task_id+1}")
                train_models["alstm"][0].add_new_module(train_models["alstm"][1])

    test_accs = [metrics.R[_NUM_TASKS-1].get(j, 0.0) for j in range(_NUM_TASKS)]
    if verbose:
        print()
        print(f"  WISDM results  --  GIM-{variant.upper()} MH (oracle routing)")
        print(f"  Validation : {[round(v,4) for v in subtask_val_accs]}")
        print(f"  Test       : {[round(t,4) for t in test_accs]}")
    return subtask_val_accs, test_accs, metrics, train_models, train_autoencoders
