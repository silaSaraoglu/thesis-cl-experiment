import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add repos to sys.path so we can import from them without installing.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT,
           os.path.join(_ROOT, 'repos', 'gim'),
           os.path.join(_ROOT, 'repos', 'esn')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


for _p in [os.path.join(_ROOT, 'single_head'),
           os.path.join(_ROOT, 'multi_head')]:
    if _p not in sys.path:
        sys.path.append(_p)


MNIST_PERM = torch.from_numpy(np.load(os.path.join(_ROOT, 'repos', 'gim', 'tasks', 'mnist', 'permutations.npy'))[0]).long()


def collect_datasets(train_cl, test_cl, task_list, max_samples, reshape=False):
    train_datasets, val_datasets, test_datasets = [], [], []
    for subset in task_list:
        train_cl.choose_subset(subset)
        loader_tr, loader_val = train_cl.get_train_val_loader(max_samples=max_samples)
        for loader, store in [(loader_tr, train_datasets), (loader_val, val_datasets)]:
            Xs, Ys = [], []
            for x, y in loader:
                if reshape:
                    x = x.reshape(x.size(0), -1)[:, MNIST_PERM].reshape(x.size(0), 28, 28)
                Xs.append(x)
                Ys.append(y)
            store.append(TensorDataset(torch.cat(Xs), torch.cat(Ys)))
        test_cl.choose_subset(subset)
        Xs, Ys = [], []
        for x, y in DataLoader(test_cl, batch_size=256, shuffle=False):
            if reshape:
                x = x.reshape(x.size(0), -1)[:, MNIST_PERM].reshape(x.size(0), 28, 28)
            Xs.append(x)
            Ys.append(y)
        test_datasets.append(TensorDataset(torch.cat(Xs), torch.cat(Ys)))
    return train_datasets, val_datasets, test_datasets


def gim_predict(train_models, x, task_id=None):
    with torch.no_grad():
        model = train_models['alstm'][0]
        model.eval()
        if task_id is not None:
            task_id = min(task_id, len(model.lstms) - 1)
        hidden_state = model.reset_memory_state(batch_size=x.size(0), module_id=task_id)
        _, preds = model(x, hidden_state, task_id=task_id)
        return preds[:, -1, :].argmax(dim=1).numpy()


def gim_kappa_over_loader(train_models, loader, transform=None):
    """Cohen's Kappa over `loader` for the GIM model, with no autoencoder routing.

    task_id=None reproduces exactly the forward path of tasks.mnist.utils_single.test
    with module_id=None -- the same call the pre-training accuracy uses -- so the
    Kappa and the accuracy describe the same predictions.

    transform: optional input permutation (px) applied by the MNIST/Fashion variants.
    """
    from shared.metrics import cohen_kappa
    all_preds, all_labels = [], []
    for x, y in loader:
        if transform is not None:
            x = transform(x)
        all_preds.extend(gim_predict(train_models, x, task_id=None))
        all_labels.extend(y.numpy())
    return cohen_kappa(all_labels, all_preds)


_SEP = "::"


def json_default(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def load_cache(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, {}
    params = {tuple(k.split(_SEP, 1)): v for k, v in payload.get("params", {}).items()}
    return params, payload.get("meta", {})


def save_cache(path, params, meta=None):
    payload = {
        "meta": meta or {},
        "params": {f"{m}{_SEP}{d}": v for (m, d), v in params.items()},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)




def load_tuned_params(path, datasets, models):
    params, _ = load_cache(path)
    return {(m, ds): params[(m, ds)] for ds in datasets for m in models}
