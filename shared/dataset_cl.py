import os
import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from sklearn.model_selection import train_test_split


def _select_torch_labels(targets, labels):
    """Return indices where a torch target tensor matches any label in labels."""
    mask = torch.stack([(targets == label) for label in labels])
    return torch.sum(mask, dim=0).nonzero().squeeze()


def _as_numpy_targets(targets, indices):
    selected = targets[indices]
    if torch.is_tensor(selected):
        return selected.cpu().numpy()
    return np.asarray(selected)


def _make_stratified_loaders(dataset, indices, targets, perc_val, batch_size):
    targets_sub = _as_numpy_targets(targets, indices)
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
        batch_size=len(val_indices),
        drop_last=False,
    )
    return DataLoader(dataset, batch_sampler=train_sampler), DataLoader(dataset, batch_sampler=val_sampler)


class ContinualDatasetMixin:
    clamp_holdout_size = False

    def _init_cl_config(self, train, perc_val, batch_size):
        self.perc_val = perc_val
        self.batch_size = batch_size
        self.train = train

    def set_holdout_config(self, holdout_n=0, holdout_seed=0, use_holdout=False):
        self._holdout_n = holdout_n
        self._holdout_seed = holdout_seed
        self._use_holdout = use_holdout

    def _active_indices(self, max_samples=None):
        indices = list(range(len(self.targets)))
        holdout_n = getattr(self, '_holdout_n', 0)
        holdout_seed = getattr(self, '_holdout_seed', 0)
        use_holdout = getattr(self, '_use_holdout', False)

        if holdout_n > 0:
            rng = np.random.RandomState(holdout_seed)
            perm = rng.permutation(len(indices)).tolist()
            if self.clamp_holdout_size:
                holdout_n = min(holdout_n, len(indices) // 2)
            indices = perm[:holdout_n] if use_holdout else perm[holdout_n:]

        if max_samples is not None and max_samples < len(indices):
            targets_sub = _as_numpy_targets(self.targets, indices)
            indices, _ = train_test_split(indices, train_size=max_samples, stratify=targets_sub)

        return indices

    def get_train_val_loader(self, max_samples=None):
        if not self.train:
            raise Exception("Cannot split train and validation when mode test is on.")
        indices = self._active_indices(max_samples=max_samples)
        return _make_stratified_loaders(self, indices, self.targets, self.perc_val, self.batch_size)


fmnist_accessory_labels = {5, 7, 8, 9}
fmnist_look_up_table = torch.zeros(10, dtype=torch.long)
for accessory in fmnist_accessory_labels:
    fmnist_look_up_table[accessory] = 1


class FashionMNIST_CL(ContinualDatasetMixin, datasets.FashionMNIST):
    """Fashion-MNIST garment(0)-vs-accessory(1) domain-incremental benchmark."""

    TASK_NAMES = {
        1: "Tshirt/Sandal",
        2: "Trouser/Sneaker",
        3: "Pullover/Ankle-boot",
        4: "Shirt/Bag",
    }

    def __init__(self, root, download, train, perc_val=0.2, batch_size=3, output_size=None):
        super().__init__(root, train=train, download=download, transform=transforms.ToTensor())
        self.all_targets = self.targets
        self.all_data = self.data
        self.output_size = output_size
        self._init_cl_config(train, perc_val, batch_size)

    def choose_subset(self, labels):
        """Select the two classes of a task; remap garment->0, accessory->1."""
        mask = _select_torch_labels(self.all_targets, labels)
        self.data = self.all_data[mask]
        self.targets = fmnist_look_up_table[self.all_targets[mask]]


# Summer (0) vs winter (1): within-category season contrasts (probe transfer gap ~0.35).
# summer = {0 T-shirt, 3 Dress, 5 Sandal} ; winter = {2 Pullover, 4 Coat, 9 Ankle-boot}
fmnist_winter_labels = {2, 4, 9}
fmnist_season_look_up_table = torch.zeros(10, dtype=torch.long)
for winter_label in fmnist_winter_labels:
    fmnist_season_look_up_table[winter_label] = 1


class FashionMNISTSeason_CL(FashionMNIST_CL):
    """Fashion-MNIST summer(0)-vs-winter(1) domain-incremental benchmark.

    Within-category season pairs (top/top, garment/garment, shoe/shoe), so the
    visual season cue differs per task -> strong cross-task shift. Reuses the same
    downloaded images as FashionMNIST_CL (only the label scheme differs)."""

    TASK_NAMES = {1: "Tshirt/Pullover", 2: "Dress/Coat", 3: "Sandal/Ankle-boot"}

    @property
    def raw_folder(self):
        # reuse FashionMNIST_CL's download (same images, different label mapping)
        return os.path.join(self.root, "FashionMNIST_CL", "raw")

    def choose_subset(self, labels):
        """Select the two classes of a task; remap summer->0, winter->1."""
        mask = _select_torch_labels(self.all_targets, labels)
        self.data = self.all_data[mask]
        self.targets = fmnist_season_look_up_table[self.all_targets[mask]]


hhar_processed_file = 'hhar_processed.npz'
hhar_raw_file = 'Phones_accelerometer.csv'
hhar_window_size = 128
hhar_window_stride = 128
hhar_max_windows_per_device_activity = 1500
hhar_test_fraction = 0.25
hhar_split_seed = 0
hhar_device_models = ['nexus4', 's3', 's3mini', 'samsungold']
hhar_activity_labels = {
    'bike': 0,
    'sit': 1,
    'stairsdown': 2,
    'stairsup': 3,
    'stand': 4,
    'walk': 5,
}


def _hhar_preprocess(root):
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
        usecols=['Creation_Time', 'x', 'y', 'z', 'Model', 'Device', 'gt'],
        dtype={'x': 'float32', 'y': 'float32', 'z': 'float32',
               'Model': 'category', 'Device': 'category', 'gt': 'category'},
    )
    df = df[df['gt'].notna() & (df['gt'].astype(str) != 'null')]

    counts = defaultdict(int)
    Xs, ys, ds = [], [], []
    for (dev, gt), g in df.groupby(['Device', 'gt'], observed=True):
        gt = str(gt)
        if gt not in hhar_activity_labels:
            continue
        model = str(g['Model'].iloc[0])
        if model not in hhar_device_models:
            continue
        m_idx = hhar_device_models.index(model)
        a_idx = hhar_activity_labels[gt]
        arr = g.sort_values('Creation_Time')[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        n = len(arr) // hhar_window_size
        for k in range(n):
            if counts[(m_idx, a_idx)] >= hhar_max_windows_per_device_activity:
                break
            w = arr[k * hhar_window_stride: k * hhar_window_stride + hhar_window_size]
            if len(w) < hhar_window_size:
                break
            Xs.append(w)
            ys.append(a_idx)
            ds.append(m_idx)
            counts[(m_idx, a_idx)] += 1

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    d = np.asarray(ds, dtype=np.int64)

    rng = np.random.RandomState(hhar_split_seed)
    tr_mask = np.zeros(len(y), dtype=bool)
    for m_idx in range(len(hhar_device_models)):
        for a_idx in hhar_activity_labels.values():
            idx = np.where((d == m_idx) & (y == a_idx))[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            n_tr = int(round(len(idx) * (1.0 - hhar_test_fraction)))
            tr_mask[idx[:n_tr]] = True

    X_tr, y_tr, d_tr = X[tr_mask], y[tr_mask], d[tr_mask]
    X_te, y_te, d_te = X[~tr_mask], y[~tr_mask], d[~tr_mask]

    mean = X_tr.mean(axis=(0, 1))
    std = X_tr.std(axis=(0, 1)) + 1e-8
    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_te = ((X_te - mean) / std).astype(np.float32)

    print(f'[HHAR] train={len(X_tr)}, test={len(X_te)}, window={hhar_window_size}')
    for m_idx, name in enumerate(hhar_device_models):
        uniq, cnts = np.unique(y_tr[d_tr == m_idx], return_counts=True)
        print(f'[HHAR]   {name}: train windows per activity = '
              f'{ {int(u): int(c) for u, c in zip(uniq, cnts)} }')

    out = os.path.join(root, hhar_processed_file)
    np.savez(out, X_train=X_tr, y_train=y_tr, d_train=d_tr,
             X_test=X_te, y_test=y_te, d_test=d_te, mean=mean, std=std)
    print(f'[HHAR] Saved -> {out}')


class HHAR_CL(ContinualDatasetMixin, torch.utils.data.Dataset):
    """HHAR phone accelerometer domain-incremental CL with a shared 6-way head."""

    clamp_holdout_size = True
    DEVICE_MODELS = hhar_device_models
    ACTIVITY_NAMES = {v: k for k, v in hhar_activity_labels.items()}
    TASK_NAMES = {i + 1: m for i, m in enumerate(hhar_device_models)}

    def __init__(self, root, train=True, download=False, perc_val=0.25, batch_size=32):
        os.makedirs(root, exist_ok=True)
        processed = os.path.join(root, hhar_processed_file)
        if not os.path.exists(processed):
            _hhar_preprocess(root)

        data = np.load(processed)
        if train:
            X, y, dom = data['X_train'], data['y_train'], data['d_train']
        else:
            X, y, dom = data['X_test'], data['y_test'], data['d_test']

        self.all_data = torch.FloatTensor(X)
        self.all_activity = y
        self.all_domain = dom
        self.all_targets = torch.LongTensor(y)

        self.data = self.all_data
        self.targets = self.all_targets
        self.activity = self.all_activity

        self._init_cl_config(train, perc_val, batch_size)

    def choose_domain(self, device_idx):
        """Select all records for one device model. Labels stay 0-5."""
        mask = (self.all_domain == int(device_idx))
        self.data = self.all_data[mask]
        self.targets = torch.LongTensor(self.all_activity[mask])

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
        self._init_cl_config(train, perc_val, batch_size)

    def choose_subset(self, labels):
        '''
        Select a subset of dataset with provided labels.

        :param labels: a list containing integer representing digits to select from dataset
        '''
        mask = _select_torch_labels(self.all_targets, labels)

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
wisdm_window_stride = 100    # 50% overlap
wisdm_balance_seed = 0   # seed for the per-class downsampling that equalises the tasks
wisdm_activity_labels = {
    'Walking':    0,
    'Jogging':    1,
    'Upstairs':   2,
    'Downstairs': 3,
    'Sitting':    4,
    'Standing':   5,
}


def _balance_per_class(X, y, seed):
    """Downsample every class to the smallest class count so the binary tasks are equal in size."""
    rng = np.random.RandomState(seed)
    _, counts = np.unique(y, return_counts=True)
    n_per_class = int(counts.min())
    keep = [rng.choice(np.where(y == c)[0], size=n_per_class, replace=False)
            for c in np.unique(y)]
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def _wisdm_preprocess(root):
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
    n_train   = max(1, int(len(all_users) * 0.8))
    train_users = set(all_users[:n_train])

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for (uid, act), samples in user_act_buffer.items():
        arr = np.array(samples, dtype=np.float32)
        i = 0
        while i + wisdm_window_size <= len(arr):
            window = arr[i:i + wisdm_window_size]
            if uid in train_users:
                X_tr.append(window); y_tr.append(act)
            else:
                X_te.append(window); y_te.append(act)
            i += wisdm_window_stride

    X_tr = np.stack(X_tr).astype(np.float32)  # (N, 200, 3)
    y_tr = np.array(y_tr, dtype=np.int64)
    X_te = np.stack(X_te).astype(np.float32)
    y_te = np.array(y_te, dtype=np.int64)

    # Equalise the activities so the three binary tasks hold the same number of samples.
    X_tr, y_tr = _balance_per_class(X_tr, y_tr, wisdm_balance_seed)
    X_te, y_te = _balance_per_class(X_te, y_te, wisdm_balance_seed)

    print(f'[WISDM] train={len(X_tr)}, test={len(X_te)}')
    uniq, counts = np.unique(y_tr, return_counts=True)
    label_names = {v: k for k, v in wisdm_activity_labels.items()}
    print(f'[WISDM] Activity distribution (train): '
          f'{ {label_names[u]: c for u, c in zip(uniq.tolist(), counts.tolist())} }')

    mean = X_tr.mean(axis=(0, 1))
    std  = X_tr.std(axis=(0, 1)) + 1e-8
    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_te = ((X_te - mean) / std).astype(np.float32)

    out = os.path.join(root, wisdm_processed_file)
    np.savez(out, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
             mean=mean, std=std)
    print(f'[WISDM] Saved -> {out}')


class WISDM_CL(ContinualDatasetMixin, torch.utils.data.Dataset):
    """WISDM accelerometer dataset — task-incremental CL.
    3 tasks, each a binary classification between two activity types.
    Interface mirrors UCI_HAR_CL for GIM.

    Labels (0-based): Walking=0, Jogging=1, Upstairs=2, Downstairs=3,
                      Sitting=4, Standing=5
    Task pairs (Option A — similar activities):
        Task 1: Walking vs Jogging      (both locomotion — hardest to separate)
        Task 2: Upstairs vs Downstairs  (both stair climbing)
        Task 3: Sitting vs Standing     (both static)
    """

    ACTIVITY_NAMES = {
        0: 'Walking', 1: 'Jogging', 2: 'Upstairs',
        3: 'Downstairs', 4: 'Sitting', 5: 'Standing',
    }
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
            _wisdm_preprocess(root)

        data = np.load(processed)
        if train:
            X, y = data['X_train'], data['y_train']
        else:
            X, y = data['X_test'],  data['y_test']

        self.all_data     = torch.FloatTensor(X)   # (N, 200, 3)
        self.all_activity = y                      # numpy (N,) raw labels 0-5
        self.all_targets  = torch.LongTensor(y)

        self.data     = self.all_data
        self.targets  = self.all_targets
        self.activity = self.all_activity

        self._init_cl_config(train, perc_val, batch_size)

    def choose_subset(self, activity_labels):
        """Select records for the given activity labels and remap to 0..len-1."""
        mask = np.isin(self.all_activity, activity_labels)
        self.data     = self.all_data[mask]
        self.activity = self.all_activity[mask]
        label_map = {lbl: i for i, lbl in enumerate(sorted(activity_labels))}
        self.targets = torch.LongTensor(
            [label_map[int(a)] for a in self.activity])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
