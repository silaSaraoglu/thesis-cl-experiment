import json
import os
from collections import defaultdict
import numpy as np


class CLMetrics:
    def __init__(self, num_tasks, joint_mode = False):
        self.num_tasks  = num_tasks
        self.joint_mode = joint_mode
        self.R        = defaultdict(dict)   # R[after_task][eval_task] = acc
        self.K        = defaultdict(dict)   # K[after_task][eval_task] = kappa
        self.pretrain = {}                  # pretrain[task_id] = acc before training task

    def record(self, after_task, eval_task, acc):
        self.R[after_task][eval_task] = float(acc)

    def record_kappa(self, after_task, eval_task, kappa):
        self.K[after_task][eval_task] = float(kappa)

    def record_pretrain(self, task_id, acc):
        self.pretrain[task_id] = acc

    def summary(self):
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
        for i in range(num_of_tasks):
            for j in range(i + 1):
                average_accuracy_over_time = average_accuracy_over_time + self.R[i].get(j, 0.0)
        average_accuracy_over_time = average_accuracy_over_time / (num_of_tasks * (num_of_tasks + 1) / 2)

        final_backward_transfer = 0.0
        for j in range(num_of_tasks - 1):
            final_backward_transfer = final_backward_transfer + self.R[num_of_tasks - 1].get(j, 0.0) - self.R[j].get(j, 0.0)
        final_backward_transfer = final_backward_transfer / (num_of_tasks - 1)

        average_backward_transfer = 0.0
        for i in range(1, num_of_tasks):
            for j in range(i):
                average_backward_transfer = average_backward_transfer + self.R[i].get(j, 0.0) - self.R[j].get(j, 0.0)
        average_backward_transfer = average_backward_transfer / (num_of_tasks * (num_of_tasks - 1) / 2)

        forward_transfer = 0.0
        for i in range(1, num_of_tasks):
            forward_transfer = forward_transfer + self.pretrain[i] - 0.5
        forward_transfer = forward_transfer / (num_of_tasks - 1)

        plasticity = 0.0
        for i in range(num_of_tasks):
            plasticity = plasticity + self.R[i].get(i, 0.0)
        plasticity = plasticity / num_of_tasks

        stability = 0.0
        for i in range(1, num_of_tasks):
            for j in range(i):
                stability = stability + self.R[i].get(j, 0.0)
        stability = stability / (num_of_tasks * (num_of_tasks - 1) / 2)

        return {
            "final_accuracy":             final_accuracy,
            "average_accuracy_over_time": average_accuracy_over_time,
            "final_backward_transfer":    final_backward_transfer,
            "average_backward_transfer":  average_backward_transfer,
            "forward_transfer":           forward_transfer,
            "final_cohen_kappa":          final_cohen_kappa,
            "plasticity":                 plasticity,
            "stability":                  stability,
        }

def cohen_kappa(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == len(y_pred):
        num_of_samples = len(y_true)
        num_of_agreements = np.sum(y_true == y_pred)
        p_o = num_of_agreements / num_of_samples
        pos_t = np.sum(y_true == 1)
        neg_t = np.sum(y_true == 0)
        pos_p = np.sum(y_pred == 1)
        neg_p = np.sum(y_pred == 0)
        p_e = (pos_t * pos_p + neg_t * neg_p) / (num_of_samples ** 2)
        if p_e == 1.0:
            return 0.0  
        kappa = (p_o - p_e) / (1 - p_e)
        return float(kappa)
    else:
        raise ValueError("Length of y_true and y_pred must be the same.")

def save_results(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to: {path}")
