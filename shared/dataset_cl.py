import os
import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from sklearn.model_selection import train_test_split


def select_torch_labels(targets, labels):
    mask = torch.stack([(targets == label) for label in labels])
    return torch.sum(mask, dim=0).nonzero().squeeze()


def as_numpy_targets(targets, indices):
    selected = targets[indices]
    if torch.is_tensor(selected):
        return selected.cpu().numpy()
    return np.asarray(selected)


def subject_val_split(indices, subjects, val_frac, val_users=None):
    idx = np.asarray(indices)
    subj = np.asarray(subjects)[idx]
    if val_users is None:
        users = np.unique(subj)                      # sorted ascending
        if len(users) < 2:
            return None
        n_val = max(1, int(round(len(users) * val_frac)))
        n_val = min(n_val, len(users) - 1)           # keep at least one training user
        val_users = set(users[-n_val:].tolist())
    val_mask = np.array([s in val_users for s in subj])
    return idx[~val_mask].tolist(), idx[val_mask].tolist()


def make_stratified_loaders(dataset, indices, targets, perc_val, batch_size):
    split = None
    if getattr(dataset, 'subject_val_split', False) and getattr(dataset, 'subjects', None) is not None:
        split = subject_val_split(indices, dataset.subjects, perc_val,
                                   val_users=getattr(dataset, 'val_users', None))
    if split is not None:
        train_indices, val_indices = split
    else:
        targets_sub = as_numpy_targets(targets, indices)
        train_indices, val_indices = train_test_split(
            indices, test_size=perc_val, shuffle=True, stratify=targets_sub)
    drop_last = len(train_indices) >= batch_size
    train_sampler = BatchSampler(
        SubsetRandomSampler(train_indices),
        batch_size=batch_size,
        drop_last=drop_last,
    )
    val_sampler = BatchSampler(
        SubsetRandomSampler(val_indices),
        batch_size=max(1, min(batch_size, len(val_indices))),
        drop_last=False,
    )
    return DataLoader(dataset, batch_sampler=train_sampler), DataLoader(dataset, batch_sampler=val_sampler)


class ContinualDatasetMixin:
    clamp_holdout_size = False

    def init_cl_config(self, train, perc_val, batch_size):
        self.perc_val = perc_val
        self.batch_size = batch_size
        self.train = train

    def set_holdout_config(self, holdout_n=0, holdout_seed=0, use_holdout=False):
        self._holdout_n = holdout_n
        self._holdout_seed = holdout_seed
        self._use_holdout = use_holdout

    def active_indices(self, max_samples=None):
        indices = list(range(len(self.targets)))
        holdout_n = getattr(self, '_holdout_n', 0)
        holdout_seed = getattr(self, '_holdout_seed', 0)
        use_holdout = getattr(self, '_use_holdout', False)

        if holdout_n > 0:
            all_idx = np.arange(len(indices))
            n = min(holdout_n, len(all_idx) // 2) if self.clamp_holdout_size else holdout_n
            n = max(1, min(n, len(all_idx) - 1))
            targets_all = as_numpy_targets(self.targets, all_idx)
            try:
                hold_idx, rest_idx = train_test_split(
                    all_idx, train_size=n, shuffle=True,
                    random_state=holdout_seed, stratify=targets_all)
            except ValueError:
                # Too few samples in some class to stratify -> plain random carve.
                rng = np.random.RandomState(holdout_seed)
                perm = rng.permutation(len(all_idx))
                hold_idx, rest_idx = perm[:n], perm[n:]
            indices = (hold_idx if use_holdout else rest_idx).tolist()

        if max_samples is not None and max_samples < len(indices):
            targets_sub = as_numpy_targets(self.targets, indices)
            indices, _ = train_test_split(indices, train_size=max_samples, stratify=targets_sub)

        return indices

    def get_train_val_loader(self, max_samples=None):
        if not self.train:
            raise Exception("Cannot split train and validation when mode test is on.")
        indices = self.active_indices(max_samples=max_samples)
        return make_stratified_loaders(self, indices, self.targets, self.perc_val, self.batch_size)


# Summer (0) vs winter (1): within-category season contrasts.
# summer = {0 T-shirt, 3 Dress, 5 Sandal} ; winter = {2 Pullover, 4 Coat, 9 Ankle-boot}
fmnist_winter_labels = {2, 4, 9}
fmnist_season_look_up_table = torch.zeros(10, dtype=torch.long)
for winter_label in fmnist_winter_labels:
    fmnist_season_look_up_table[winter_label] = 1


class FashionMNISTSeason_CL(ContinualDatasetMixin, datasets.FashionMNIST):
    """Fashion-MNIST summer(0)-vs-winter(1) domain-incremental benchmark.

    Within-category season pairs (top/top, garment/garment, shoe/shoe), so the
    visual season cue differs per task -> strong cross-task shift."""

    TASK_NAMES = {1: "Tshirt/Pullover", 2: "Dress/Coat", 3: "Sandal/Ankle-boot"}

    def __init__(self, root, download, train, perc_val=0.2, batch_size=3, output_size=None):
        super().__init__(root, train=train, download=download, transform=transforms.ToTensor())
        self.all_targets = self.targets
        self.all_data = self.data
        self.output_size = output_size
        self.init_cl_config(train, perc_val, batch_size)

    @property
    def raw_folder(self):
        # downloaded Fashion-MNIST images live under this folder name on disk
        return os.path.join(self.root, "FashionMNIST_CL", "raw")

    def choose_subset(self, labels):
        """Select the two classes of a task; remap summer->0, winter->1."""
        mask = select_torch_labels(self.all_targets, labels)
        self.data = self.all_data[mask]
        self.targets = fmnist_season_look_up_table[self.all_targets[mask]]


hhar_processed_file = 'hhar_processed.npz'
hhar_raw_file = 'Phones_accelerometer.csv'
hhar_window_size = 128
hhar_window_stride = 128
hhar_max_windows_per_device_activity = 1500
# Subject-independent train/test split fraction, shared by both sensor datasets
# (HHAR and WISDM): 0.8 of users -> train, the rest -> test.
sensor_train_fraction = 0.8
hhar_n_val_users = 1     # pool users (highest ids) held out for validation
hhar_balance_seed = 0    # seed for the per-(device, activity) downsampling
hhar_device_models = ['nexus4', 's3', 's3mini', 'samsungold']
hhar_activity_labels = {
    'bike': 0,
    'sit': 1,
    'stairsdown': 2,
    'stairsup': 3,
    'stand': 4,
    'walk': 5,
}


def balance_hhar(X, y, d, u, seed):
    """Downsample every (device, activity) cell to the smallest such cell.

    After this, within each device the 6 activities hold an equal number of windows
    AND all four device tasks hold the same total (6 x min-cell) -> the per-class and
    the per-task imbalance are both removed at once. Operates on one split at a time
    (train / val / test), so no window crosses a user boundary and the split stays
    subject-independent. Returns the balanced arrays and the per-cell count kept."""
    rng = np.random.RandomState(seed)
    devices = np.unique(d)
    activities = np.unique(y)
    n_cell = min(int(np.sum((d == dev) & (y == act)))
                 for dev in devices for act in activities)
    keep = np.sort(np.concatenate([
        rng.choice(np.where((d == dev) & (y == act))[0], size=n_cell, replace=False)
        for dev in devices for act in activities]))
    return X[keep], y[keep], d[keep], u[keep], n_cell


def hhar_preprocess(root):
    """Preprocess HHAR phone accelerometer data into train/test device tasks."""
    import pandas as pd
    from collections import defaultdict

    raw_path = os.path.join(root, hhar_raw_file)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f'HHAR raw file not found at {raw_path}.\n'
            f'Download the "Heterogeneity Activity Recognition" zip from '
            f'https://archive.ics.uci.edu/dataset/344 and place '
            f'Phones_accelerometer.csv in {root}')

    df = pd.read_csv(
        raw_path,
        usecols=['Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'gt'],
        dtype={'x': 'float32', 'y': 'float32', 'z': 'float32',
               'User': 'category', 'Model': 'category', 'gt': 'category'},
    )
    df = df[df['gt'].notna() & (df['gt'].astype(str) != 'null')]

    counts = defaultdict(int)
    Xs, ys, ds, us = [], [], [], []
    for (user, model, gt), g in df.groupby(['User', 'Model', 'gt'], observed=True):
        gt = str(gt); model = str(model); user = str(user)
        if gt not in hhar_activity_labels:
            continue
        if model not in hhar_device_models:
            continue
        m_idx = hhar_device_models.index(model)
        a_idx = hhar_activity_labels[gt]
        arr = g.sort_values('Creation_Time')[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        n = len(arr) // hhar_window_size
        for k in range(n):
            if counts[(user, m_idx, a_idx)] >= hhar_max_windows_per_device_activity:
                break
            w = arr[k * hhar_window_stride: k * hhar_window_stride + hhar_window_size]
            if len(w) < hhar_window_size:
                break
            Xs.append(w)
            ys.append(a_idx)
            ds.append(m_idx)
            us.append(user)
            counts[(user, m_idx, a_idx)] += 1

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    d = np.asarray(ds, dtype=np.int64)

    # Subject-independent split: hold out whole users for test (same scheme as WISDM).
    # 80% of users -> train+val pool, the rest -> test; users encoded 0..n-1 in sorted
    # order. The pool's highest-id user(s) are the validation subject(s) (matches
    # HHAR_CL.n_val_users) so train / val / test are three disjoint subject sets.
    unique_users = sorted(set(us))
    user_code = {u: i for i, u in enumerate(unique_users)}
    u = np.asarray([user_code[v] for v in us], dtype=np.int64)
    n_pool_u = max(1, int(round(len(unique_users) * sensor_train_fraction)))
    n_train_u = max(1, n_pool_u - hhar_n_val_users)
    train_mask = u < n_train_u
    val_mask = (u >= n_train_u) & (u < n_pool_u)
    test_mask = u >= n_pool_u

    # Balance each split on its own: equalises the 6 activities within every device and
    # the four device tasks against each other, identically for train / val / test, so
    # train and test stay directly comparable (chance = 1/6 on every task in both).
    X_tr, y_tr, d_tr, u_tr, n_tr_cell = balance_hhar(
        X[train_mask], y[train_mask], d[train_mask], u[train_mask], hhar_balance_seed)
    X_va, y_va, d_va, u_va, n_va_cell = balance_hhar(
        X[val_mask], y[val_mask], d[val_mask], u[val_mask], hhar_balance_seed)
    X_te, y_te, d_te, u_te, n_te_cell = balance_hhar(
        X[test_mask], y[test_mask], d[test_mask], u[test_mask], hhar_balance_seed)

    # Normalise with statistics from the training split only (no val/test leakage).
    mean = X_tr.mean(axis=(0, 1))
    std = X_tr.std(axis=(0, 1)) + 1e-8
    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_va = ((X_va - mean) / std).astype(np.float32)
    X_te = ((X_te - mean) / std).astype(np.float32)

    # The dataset class carves validation back out by user id, so the pool (train+val)
    # is stored together under the *_train keys, each part already balanced.
    X_tr = np.concatenate([X_tr, X_va]).astype(np.float32)
    y_tr = np.concatenate([y_tr, y_va])
    d_tr = np.concatenate([d_tr, d_va])
    u_tr = np.concatenate([u_tr, u_va])

    print(f'[HHAR] users={len(unique_users)} (train {n_train_u} / val {n_pool_u - n_train_u} / '
          f'test {len(unique_users) - n_pool_u}), window={hhar_window_size}')
    print(f'[HHAR] balanced per (device, activity) cell -> train={n_tr_cell}, val={n_va_cell}, '
          f'test={n_te_cell} windows/cell')
    print(f'[HHAR] per-task windows: train={6 * n_tr_cell}/device, val={6 * n_va_cell}/device, '
          f'test={6 * n_te_cell}/device | totals train={len(X_tr)}, test={len(X_te)}')

    out = os.path.join(root, hhar_processed_file)
    np.savez(out, X_train=X_tr, y_train=y_tr, d_train=d_tr, u_train=u_tr,
             X_test=X_te, y_test=y_te, d_test=d_te, u_test=u_te, mean=mean, std=std)
    print(f'[HHAR] Saved -> {out}')


class HHAR_CL(ContinualDatasetMixin, torch.utils.data.Dataset):
    """HHAR phone accelerometer domain-incremental CL with a shared 6-way head."""

    clamp_holdout_size = True
    subject_val_split = True   # hold out whole users for validation (subject-independent)
    n_val_users = hhar_n_val_users  # train-pool user(s) (highest id) held out for validation;
                               # the rest train, the 2 preprocessing-held users are test
    DEVICE_MODELS = hhar_device_models
    TASK_NAMES = {i + 1: m for i, m in enumerate(hhar_device_models)}

    def __init__(self, root, train=True, download=False, perc_val=0.25, batch_size=32):
        os.makedirs(root, exist_ok=True)
        processed = os.path.join(root, hhar_processed_file)
        if not os.path.exists(processed):
            hhar_preprocess(root)

        data = np.load(processed)
        if train:
            X, y, dom, usr = data['X_train'], data['y_train'], data['d_train'], data['u_train']
        else:
            X, y, dom, usr = data['X_test'], data['y_test'], data['d_test'], data['u_test']

        self.all_data = torch.FloatTensor(X)
        self.all_activity = y
        self.all_domain = dom
        self.all_user = np.asarray(usr)
        self.all_targets = torch.LongTensor(y)

        self.data = self.all_data
        self.targets = self.all_targets
        self.activity = self.all_activity
        self.subjects = self.all_user

        self.init_cl_config(train, perc_val, batch_size)
        # Fixed subject-independent validation split: the highest-id train-pool user(s)
        # are the validation subject(s); the rest are train (test users are already held
        # out at preprocessing). Deterministic and identical across tuning/final and every
        # device -> a clean fixed 3-way subject partition. Only the train split needs it.
        self.val_users = (set(sorted(np.unique(self.all_user).tolist())[-self.n_val_users:])
                          if train else None)

    def choose_domain(self, device_idx):
        """Select all records for one device model. Labels stay 0-5."""
        mask = (self.all_domain == int(device_idx))
        self.data = self.all_data[mask]
        self.targets = torch.LongTensor(self.all_activity[mask])
        self.subjects = self.all_user[mask]

    def choose_subset(self, device_idx):
        if isinstance(device_idx, (list, tuple, np.ndarray)):
            device_idx = int(device_idx[0])
        self.choose_domain(device_idx)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MNIST_CL(ContinualDatasetMixin, datasets.MNIST):
    def __init__(self, root, download, train, perc_val=0.2, batch_size=3, output_size=None):
        '''
        :param output_size: number of output units of the model
        '''
        super(MNIST_CL, self).__init__(root, train=train, download=download, transform=transforms.ToTensor())

        self.all_targets = self.targets
        self.all_data = self.data
        self.output_size = output_size
        self.init_cl_config(train, perc_val, batch_size)

    def choose_subset(self, labels):
        '''
        Select a subset of dataset with provided labels.

        :param labels: a list containing integer representing digits to select from dataset
        '''
        mask = select_torch_labels(self.all_targets, labels)

        self.targets = self.all_targets[mask]
        self.data = self.all_data[mask]

        # restrict output targets to output_size values
        # (even digits -> 0, odd digits -> 1; all task pairs are mixed-parity)
        if self.output_size is not None:
            self.targets = self.targets % self.output_size

# ══════════════════════════════════════════════════════════════════════════════
#  WISDM  —  Wireless Sensor Data Mining (accelerometer)
#  3 tasks: Walking/Jogging · Upstairs/Downstairs · Sitting/Standing
#  Labels (0-based): Walking=0, Jogging=1, Upstairs=2, Downstairs=3,
#                    Sitting=4, Standing=5
# ══════════════════════════════════════════════════════════════════════════════

wisdm_processed_file = 'wisdm_processed.npz'
wisdm_raw_file = 'WISDM_ar_v1.1_raw.txt'
wisdm_window_size = 200   # 10 s at 20 Hz (Azghan et al., CL-for-HAR convention)
wisdm_window_stride = 200    # non-overlapping windows (like HHAR) — no temporal leakage
wisdm_balance_seed = 0   # seed for the per-class downsampling that equalises the tasks
wisdm_activity_labels = {
    'Walking':    0,
    'Jogging':    1,
    'Upstairs':   2,
    'Downstairs': 3,
    'Sitting':    4,
    'Standing':   5,
}


def balance_indices(y, seed):
    """Indices that downsample every class to the smallest class count (equal-size tasks)."""
    rng = np.random.RandomState(seed)
    _, counts = np.unique(y, return_counts=True)
    n_per_class = int(counts.min())
    keep = [rng.choice(np.where(y == c)[0], size=n_per_class, replace=False)
            for c in np.unique(y)]
    return np.sort(np.concatenate(keep))


def wisdm_preprocess(root):
    """Parse WISDM raw file, segment into windows, split by user, balance classes, z-score normalise."""
    raw_path = os.path.join(root, wisdm_raw_file)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f'WISDM raw file not found at {raw_path}.\n'
            f'Download WISDM_ar_v1.1_raw.txt from '
            f'https://www.cis.fordham.edu/wisdm/dataset.php and place it in {root}')

    rows = []
    with open(raw_path, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                continue
            user_id, activity = parts[0].strip(), parts[1].strip()
            try:
                x_val = float(parts[3].strip())
                y_val = float(parts[4].strip())
                z_val = float(parts[5].strip())
            except ValueError:
                continue
            act_code = wisdm_activity_labels.get(activity)
            if act_code is None:
                continue
            rows.append((int(user_id), act_code, x_val, y_val, z_val))

    # group by (user, activity) then slide windows
    from collections import defaultdict
    user_act_buffer = defaultdict(list)
    for uid, act, x, y, z in rows:
        user_act_buffer[(uid, act)].append([x, y, z])

    all_users = sorted(set(uid for uid, _ in user_act_buffer))
    n_train   = max(1, int(len(all_users) * sensor_train_fraction))
    train_users = set(all_users[:n_train])

    X_tr, y_tr, u_tr, X_te, y_te, u_te = [], [], [], [], [], []
    for (uid, act), samples in user_act_buffer.items():
        arr = np.array(samples, dtype=np.float32)
        i = 0
        while i + wisdm_window_size <= len(arr):
            window = arr[i:i + wisdm_window_size]
            if uid in train_users:
                X_tr.append(window); y_tr.append(act); u_tr.append(uid)
            else:
                X_te.append(window); y_te.append(act); u_te.append(uid)
            i += wisdm_window_stride

    X_tr = np.stack(X_tr).astype(np.float32)  # (N, 200, 3)
    y_tr = np.array(y_tr, dtype=np.int64)
    u_tr = np.array(u_tr, dtype=np.int64)
    X_te = np.stack(X_te).astype(np.float32)
    y_te = np.array(y_te, dtype=np.int64)
    u_te = np.array(u_te, dtype=np.int64)

    # Equalise the activities so the three binary tasks hold the same number of samples
    # (keep user ids aligned with the kept windows).
    keep_tr = balance_indices(y_tr, wisdm_balance_seed)
    X_tr, y_tr, u_tr = X_tr[keep_tr], y_tr[keep_tr], u_tr[keep_tr]
    keep_te = balance_indices(y_te, wisdm_balance_seed)
    X_te, y_te, u_te = X_te[keep_te], y_te[keep_te], u_te[keep_te]

    print(f'[WISDM] users: train {len(train_users)} / test {len(all_users) - len(train_users)} '
          f'| train_windows={len(X_tr)}, test_windows={len(X_te)}')
    uniq, counts = np.unique(y_tr, return_counts=True)
    label_names = {v: k for k, v in wisdm_activity_labels.items()}
    print(f'[WISDM] Activity distribution (train): '
          f'{ {label_names[u]: c for u, c in zip(uniq.tolist(), counts.tolist())} }')

    mean = X_tr.mean(axis=(0, 1))
    std  = X_tr.std(axis=(0, 1)) + 1e-8
    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_te = ((X_te - mean) / std).astype(np.float32)

    out = os.path.join(root, wisdm_processed_file)
    np.savez(out, X_train=X_tr, y_train=y_tr, u_train=u_tr,
             X_test=X_te, y_test=y_te, u_test=u_te, mean=mean, std=std)
    print(f'[WISDM] Saved -> {out}')


class WISDM_CL(ContinualDatasetMixin, torch.utils.data.Dataset):
    """WISDM accelerometer dataset — task-incremental CL.
    3 tasks, each a binary classification between two activity types.

    Labels (0-based): Walking=0, Jogging=1, Upstairs=2, Downstairs=3,
                      Sitting=4, Standing=5
    Task pairs (similar activities):
        Task 1: Walking vs Jogging      (both locomotion — hardest to separate)
        Task 2: Upstairs vs Downstairs  (both stair climbing)
        Task 3: Sitting vs Standing     (both static)
    """

    subject_val_split = True   # hold out whole users for validation (subject-independent)

    TASK_NAMES = {
        1: 'Walking/Jogging',
        2: 'Upstairs/Downstairs',
        3: 'Sitting/Standing',
    }

    def __init__(self, root, train=True, download=False,
                 perc_val=0.25, batch_size=32):
        os.makedirs(root, exist_ok=True)
        processed = os.path.join(root, wisdm_processed_file)

        if not os.path.exists(processed):
            wisdm_preprocess(root)

        data = np.load(processed)
        if train:
            X, y, usr = data['X_train'], data['y_train'], data['u_train']
        else:
            X, y, usr = data['X_test'],  data['y_test'],  data['u_test']

        self.all_data     = torch.FloatTensor(X)   # (N, 200, 3)
        self.all_activity = y                      # numpy (N,) raw labels 0-5
        self.all_user     = np.asarray(usr)        # numpy (N,) user id per window
        self.all_targets  = torch.LongTensor(y)

        self.data     = self.all_data
        self.targets  = self.all_targets
        self.activity = self.all_activity
        self.subjects = self.all_user

        self.init_cl_config(train, perc_val, batch_size)

    def choose_subset(self, activity_labels):
        """Select records for the given activity labels and remap to 0..len-1.
        ``subjects`` is restricted to the same windows so the validation split
        holds out whole users (subject-independent)."""
        mask = np.isin(self.all_activity, activity_labels)
        self.data     = self.all_data[mask]
        self.activity = self.all_activity[mask]
        self.subjects = self.all_user[mask]
        label_map = {lbl: i for i, lbl in enumerate(sorted(activity_labels))}
        self.targets = torch.LongTensor(
            [label_map[int(a)] for a in self.activity])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
