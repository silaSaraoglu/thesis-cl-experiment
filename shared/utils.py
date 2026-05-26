"""
Shared utilities for single_head and multi_head experiments.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset


def collect_datasets(train_cl, test_cl, task_list, max_samples, batch_size=None, reshape=False):
    """Materialise train/val/test TensorDatasets for all tasks upfront.

    reshape=True removes the channel dim: (B, 1, H, W) → (B, H, W) — H time steps × W features.
    """
    train_datasets, val_datasets, test_datasets = [], [], []
    for subset in task_list:
        train_cl.choose_subset(subset)
        loader_tr, loader_val = train_cl.get_train_val_loader(max_samples=max_samples)
        for loader, store in [(loader_tr, train_datasets), (loader_val, val_datasets)]:
            Xs, Ys = [], []
            for x, y in loader:
                if reshape:
                    x = x.squeeze(1)
                Xs.append(x)
                Ys.append(y)
            store.append(TensorDataset(torch.cat(Xs), torch.cat(Ys)))
        test_cl.choose_subset(subset)
        Xs, Ys = [], []
        for x, y in DataLoader(test_cl, batch_size=256, shuffle=False):
            if reshape:
                x = x.squeeze(1)
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
