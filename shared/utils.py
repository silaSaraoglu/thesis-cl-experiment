"""
Shared utilities for single_head and multi_head experiments.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load GIM paper permutation from the original repo (first of 10 pre-saved permutations).
# The GIM paper permutes the 784 pixels with a fixed random shuffle. Here the permuted
# pixels are reshaped back to (B, 28, 28) so every model keeps the 28-step x 28-feature
# sequence format (input_size stays 28).
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, 'repos')):
    _ROOT = os.path.dirname(_ROOT)
MNIST_PERM = torch.from_numpy(
    np.load(os.path.join(_ROOT, 'repos', 'gim', 'tasks', 'mnist', 'permutations.npy'))[0]
).long()


def permute_mnist(x):
    """Apply the GIM pixel permutation, keeping the (B, 28, 28) shape.

    x: (B, 1, 28, 28) or (B, 28, 28) -> (B, 28, 28) with pixels shuffled by MNIST_PERM.
    """
    return x.reshape(x.size(0), -1)[:, MNIST_PERM].reshape(x.size(0), 28, 28)


def collect_datasets(train_cl, test_cl, task_list, max_samples, batch_size=None, reshape=False):
    """Materialise train/val/test TensorDatasets for all tasks upfront.

    reshape=True  : permuted MNIST (B, 1, 28, 28) -> (B, 28, 28) — 28 steps x 28 features
    reshape=False : no transformation (WISDM)
    """
    train_datasets, val_datasets, test_datasets = [], [], []
    for subset in task_list:
        train_cl.choose_subset(subset)
        loader_tr, loader_val = train_cl.get_train_val_loader(max_samples=max_samples)
        for loader, store in [(loader_tr, train_datasets), (loader_val, val_datasets)]:
            Xs, Ys = [], []
            for x, y in loader:
                if reshape:
                    x = permute_mnist(x)
                Xs.append(x)
                Ys.append(y)
            store.append(TensorDataset(torch.cat(Xs), torch.cat(Ys)))
        test_cl.choose_subset(subset)
        Xs, Ys = [], []
        for x, y in DataLoader(test_cl, batch_size=256, shuffle=False):
            if reshape:
                x = permute_mnist(x)
            Xs.append(x)
            Ys.append(y)
        test_datasets.append(TensorDataset(torch.cat(Xs), torch.cat(Ys)))
    return train_datasets, val_datasets, test_datasets


def gim_predict(train_models, x, task_id=None):
    """
    One forward pass through the alstm model.
    x must already be on device and correctly shaped (B, seq_len, features).
    Returns predicted class labels as numpy array of shape (B,).
    """
    with torch.no_grad():
        model = train_models['alstm'][0]
        model.eval()
        if task_id is not None:
            task_id = min(task_id, len(model.lstms) - 1)
        hidden_state = model.reset_memory_state(batch_size=x.size(0), module_id=task_id)
        _, preds = model(x, hidden_state, task_id=task_id)
        return preds[:, -1, :].argmax(dim=1).numpy()
