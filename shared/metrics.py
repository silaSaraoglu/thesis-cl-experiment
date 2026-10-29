import json
import os
from collections import defaultdict
import numpy as np


class CLMetrics:
    def __init__(self, num_tasks, joint_mode = False):
        self.num_tasks  = num_tasks
        self.joint_mode = joint_mode
        self.R          = defaultdict(dict)
        self.K          = defaultdict(dict)
        self.pretrain   = {}
        self.pretrain_K = {}

    def record(self, after_task, eval_task, acc):
        self.R[after_task][eval_task] = float(acc)

    def record_kappa(self, after_task, eval_task, kappa):
        self.K[after_task][eval_task] = float(kappa)

    def record_pretrain(self, task_id, acc):
        self.pretrain[task_id] = float(acc)

    def record_pretrain_kappa(self, task_id, kappa):
        """Zero-shot Kappa on task `task_id` before it is trained on (for Kappa-FWT).

        Chance-level performance is kappa = 0 by construction, so callers that cannot
        evaluate the upcoming task at all (multi-head models have no head for it yet)
        record 0.0 -- the exact analogue of recording `chance` for the accuracy FWT.
        """
        self.pretrain_K[task_id] = float(kappa)

    def summary(self, chance=0.5):
        num_of_tasks = self.num_tasks

        final_accuracy = 0.0
        final_cohen_kappa = 0.0
        for j in range(num_of_tasks):
            final_accuracy = final_accuracy + self.R[num_of_tasks - 1].get(j, 0.0)
            final_cohen_kappa = final_cohen_kappa + self.K[num_of_tasks - 1].get(j, 0.0)

        final_accuracy = final_accuracy/num_of_tasks
        final_cohen_kappa = final_cohen_kappa/num_of_tasks
        if self.joint_mode:
            return {
                "final_accuracy":    final_accuracy,
                "final_cohen_kappa": final_cohen_kappa,
            }

        average_accuracy_over_time = 0.0
        average_kappa_over_time = 0.0
        for i in range(num_of_tasks):
            for j in range(i + 1):
                average_accuracy_over_time = average_accuracy_over_time + self.R[i].get(j, 0.0)
                average_kappa_over_time = average_kappa_over_time + self.K[i].get(j, 0.0)
        average_accuracy_over_time = average_accuracy_over_time / (num_of_tasks * (num_of_tasks + 1) / 2)
        average_kappa_over_time = average_kappa_over_time / (num_of_tasks * (num_of_tasks + 1) / 2)

        final_backward_transfer = 0.0
        final_backward_transfer_kappa = 0.0
        for j in range(num_of_tasks - 1):
            final_backward_transfer = final_backward_transfer + self.R[num_of_tasks - 1].get(j, 0.0) - self.R[j].get(j, 0.0)
            final_backward_transfer_kappa = final_backward_transfer_kappa + self.K[num_of_tasks - 1].get(j, 0.0) - self.K[j].get(j, 0.0)
        final_backward_transfer = final_backward_transfer / (num_of_tasks - 1)
        final_backward_transfer_kappa = final_backward_transfer_kappa / (num_of_tasks - 1)

        average_backward_transfer = 0.0
        average_backward_transfer_kappa = 0.0
        for i in range(1, num_of_tasks):
            for j in range(i):
                average_backward_transfer = average_backward_transfer + self.R[i].get(j, 0.0) - self.R[j].get(j, 0.0)
                average_backward_transfer_kappa = average_backward_transfer_kappa + self.K[i].get(j, 0.0) - self.K[j].get(j, 0.0)
        average_backward_transfer = average_backward_transfer / (num_of_tasks * (num_of_tasks - 1) / 2)
        average_backward_transfer_kappa = average_backward_transfer_kappa / (num_of_tasks * (num_of_tasks - 1) / 2)

        forward_transfer = 0.0
        for i in range(1, num_of_tasks):
            forward_transfer = forward_transfer + self.pretrain.get(i, chance) - chance
        forward_transfer = forward_transfer / (num_of_tasks - 1)

        # Kappa is already chance-corrected (kappa = 0 *is* chance agreement), so the
        # accuracy version's "- chance" baseline term is identically 0 here and the
        # metric reduces to the mean zero-shot Kappa on the not-yet-trained tasks.
        forward_transfer_kappa = 0.0
        for i in range(1, num_of_tasks):
            forward_transfer_kappa = forward_transfer_kappa + self.pretrain_K.get(i, 0.0)
        forward_transfer_kappa = forward_transfer_kappa / (num_of_tasks - 1)

        plasticity = 0.0
        plasticity_kappa = 0.0
        for i in range(num_of_tasks):
            plasticity = plasticity + self.R[i].get(i, 0.0)
            plasticity_kappa = plasticity_kappa + self.K[i].get(i, 0.0)
        plasticity = plasticity / num_of_tasks
        plasticity_kappa = plasticity_kappa / num_of_tasks

        stability = 0.0
        stability_kappa = 0.0
        for i in range(1, num_of_tasks):
            for j in range(i):
                stability = stability + self.R[i].get(j, 0.0)
                stability_kappa = stability_kappa + self.K[i].get(j, 0.0)
        stability = stability / (num_of_tasks * (num_of_tasks - 1) / 2)
        stability_kappa = stability_kappa / (num_of_tasks * (num_of_tasks - 1) / 2)

        return {
            "final_accuracy":             final_accuracy,
            "average_accuracy_over_time": average_accuracy_over_time,
            "final_backward_transfer":    final_backward_transfer,
            "average_backward_transfer":  average_backward_transfer,
            "forward_transfer":           forward_transfer,
            "plasticity":                 plasticity,
            "stability":                  stability,
            # Kappa-valued counterparts of every metric above, computed from the
            # task x task Kappa matrix self.K exactly as the accuracy suite is
            # computed from self.R.
            "final_cohen_kappa":                final_cohen_kappa,
            "average_kappa_over_time":          average_kappa_over_time,
            "final_backward_transfer_kappa":    final_backward_transfer_kappa,
            "average_backward_transfer_kappa":  average_backward_transfer_kappa,
            "forward_transfer_kappa":           forward_transfer_kappa,
            "plasticity_kappa":                 plasticity_kappa,
            "stability_kappa":                  stability_kappa,
        }

def cohen_kappa(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_of_samples = len(y_true)
    if num_of_samples == 0:
        return 0.0
    p_o = np.sum(y_true == y_pred) / num_of_samples
    p_e = 0.0
    for c in np.union1d(y_true, y_pred):
        p_e += np.sum(y_true == c) * np.sum(y_pred == c)
    p_e /= num_of_samples ** 2
    if p_e == 1.0:
        return 0.0
    return float((p_o - p_e) / (1 - p_e))

def kappa_from_predict(model, loader):
    """Cohen's Kappa over `loader` for models exposing .predict (the ESN family).

    Mirrors the prediction path of those models' own calculate_accuracy /
    R-matrix eval loop, so the Kappa and the accuracy describe the same forward pass.
    """
    all_preds, all_labels = [], []
    for x, y in loader:
        all_preds.extend(model.predict(x))
        all_labels.extend(y.numpy())
    return cohen_kappa(all_labels, all_preds)


def kappa_from_logits(model, loader, transform=None):
    """Cohen's Kappa over `loader` for torch classifiers scored by argmax (LSTM family).

    transform: optional input permutation -- the px() the MNIST/Fashion variants apply
    (HHAR passes None). Restores the model's original train/eval mode on the way out.
    """
    import torch
    all_preds, all_labels = [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            if transform is not None:
                x = transform(x)
            all_preds.extend(model(x).argmax(dim=1).numpy())
            all_labels.extend(y.numpy())
    if was_training:
        model.train()
    return cohen_kappa(all_labels, all_preds)


def save_results(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to: {path}")
