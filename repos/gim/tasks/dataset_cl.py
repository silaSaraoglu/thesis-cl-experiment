import random as _random
import os

import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from sklearn.model_selection import train_test_split

class MNIST_CL(datasets.MNIST):
    def __init__(self, root, download, train, perc_val=0.2, batch_size=3, output_size=None, image_size=28):
        '''
        :param output_size: number of output units of the model
        :param image_size: resize images to image_size×image_size before converting to tensor
        '''
        if image_size != 28:
            tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        else:
            tf = transforms.ToTensor()
        super(MNIST_CL, self).__init__(root, train=train, download=download, transform=tf)

        self.all_targets = self.targets
        self.all_data = self.data
        
        self.perc_val = perc_val
        self.batch_size = batch_size

        self.output_size = output_size

        self.train = train
    
    def choose_subset(self, labels):
        '''
        Select a subset of dataset with provided labels.

        :param labels: a list containing integer representing digits to select from dataset
        '''


        mask = torch.stack([ (self.all_targets == l) for l in labels ]) # one mask for each label
        mask = torch.sum(mask, dim=0) # or of masks
        mask = mask.nonzero().squeeze() # index of valid elements

        self.targets = self.all_targets[mask]
        self.data = self.all_data[mask]

        # restrict output targets to output_size values
        if self.output_size is not None:
            self.targets = self.targets % self.output_size
    
    def get_train_val_loader(self, max_samples=None):
        '''
        :return mnist_cl_loader_train: mini-batch loader for training set
        :return mnist_cl_loader_val:   mini-batch loader for validation set
        '''
        if self.train:
            indices = list(range(self.targets.size(0)))
            if max_samples is not None and max_samples < len(indices):
                _tgt = self.targets[indices].numpy()
                indices, _ = train_test_split(indices, train_size=max_samples, stratify=_tgt)
            targets_sub = self.targets[indices].numpy()
            train_indices, val_indices = train_test_split(
                indices, test_size=self.perc_val, shuffle=True, stratify=targets_sub)
            train_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size=self.batch_size, drop_last=True)
            val_sampler   = BatchSampler(SubsetRandomSampler(val_indices),   batch_size=len(val_indices), drop_last=False)
            mnist_cl_loader_train = DataLoader(self, batch_sampler=train_sampler)
            mnist_cl_loader_val   = DataLoader(self, batch_sampler=val_sampler)
            return mnist_cl_loader_train, mnist_cl_loader_val
        else:
            raise Exception("Cannot split train and validation when mode test is on.")


# ══════════════════════════════════════════════════════════════════════════════
#  UCI HAR  —  Human Activity Recognition Using Smartphones
#  3 tasks: [WALKING, WALKING_UP], [WALKING_DOWN, SITTING], [STANDING, LAYING]
# ══════════════════════════════════════════════════════════════════════════════

_UCIHAR_SIGNALS = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z',
]
_UCIHAR_PROCESSED = 'ucihar_processed.npz'


def _ucihar_download(root):
    """Download and extract UCI HAR Dataset zip."""
    import zipfile
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests  -- needed for automatic download")

    zip_path = os.path.join(root, 'UCI_HAR_Dataset.zip')
    if not os.path.exists(zip_path):
        url = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
               '/00240/UCI%20HAR%20Dataset.zip')
        print('[UCI-HAR] Downloading dataset ...')
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)

    har_dir = os.path.join(root, 'UCI HAR Dataset')
    if not os.path.isdir(har_dir):
        print('[UCI-HAR] Extracting ...')
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)


def _ucihar_load_split(har_dir, split):
    """Load raw inertial signals for 'train' or 'test'."""
    signals_dir = os.path.join(har_dir, split, 'Inertial Signals')
    channels = []
    for sig in _UCIHAR_SIGNALS:
        fname = os.path.join(signals_dir, f'{sig}_{split}.txt')
        channels.append(np.loadtxt(fname))          # (N, 128)
    X = np.stack(channels, axis=-1).astype(np.float32)  # (N, 128, 9)
    y = np.loadtxt(os.path.join(har_dir, split, f'y_{split}.txt'),
                   dtype=np.int64)                  # (N,)  values 1-6
    return X, y


def _ucihar_preprocess(root):
    """z-score raw signals and save .npz."""
    har_dir = os.path.join(root, 'UCI HAR Dataset')
    if not os.path.isdir(har_dir):
        raise FileNotFoundError(
            f'UCI HAR Dataset folder not found in {root}. '
            f'Pass download=True or place the extracted folder there.')

    X_tr, y_tr = _ucihar_load_split(har_dir, 'train')
    X_te, y_te = _ucihar_load_split(har_dir, 'test')
    print(f'[UCI-HAR] train={len(X_tr)}, test={len(X_te)}')
    uniq, counts = np.unique(y_tr, return_counts=True)
    print(f'[UCI-HAR] Activity distribution (train): {dict(zip(uniq.tolist(), counts.tolist()))}')

    mean = X_tr.mean(axis=(0, 1))
    std  = X_tr.std(axis=(0, 1)) + 1e-8
    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_te = ((X_te - mean) / std).astype(np.float32)

    out = os.path.join(root, _UCIHAR_PROCESSED)
    np.savez(out, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
             mean=mean, std=std)
    print(f'[UCI-HAR] Saved -> {out}')


class UCI_HAR_CL(torch.utils.data.Dataset):
    """UCI HAR — task-incremental CL dataset.
    3 tasks, each a binary classification between two activity types.
    Interface mirrors PhysioNet_CL / MNIST_CL for GIM.
    """

    ACTIVITY_NAMES = {
        1: 'WALKING', 2: 'WALKING_UP', 3: 'WALKING_DOWN',
        4: 'SITTING', 5: 'STANDING',  6: 'LAYING',
    }
    TASK_NAMES = {
        1: 'WALKING/SITTING',
        2: 'WALKING_UP/STANDING',
        3: 'WALKING_DOWN/LAYING',
    }

    def __init__(self, root, train=True, download=False,
                 perc_val=0.25, batch_size=32):
        os.makedirs(root, exist_ok=True)
        processed = os.path.join(root, _UCIHAR_PROCESSED)

        if not os.path.exists(processed):
            if download:
                _ucihar_download(root)
            _ucihar_preprocess(root)

        data = np.load(processed)
        if train:
            X, y = data['X_train'], data['y_train']
        else:
            X, y = data['X_test'],  data['y_test']

        self.all_data     = torch.FloatTensor(X)        # (N, 128, 9)
        self.all_activity = y                           # numpy (N,) raw labels 1-6
        self.all_targets  = torch.LongTensor(y)         # placeholder, remapped in choose_subset

        self.data     = self.all_data
        self.targets  = self.all_targets
        self.activity = self.all_activity

        self.perc_val   = perc_val
        self.batch_size = batch_size
        self.train      = train

    def choose_subset(self, activity_labels):
        """Select records for the given activity labels and remap to 0..len-1."""
        mask = np.isin(self.all_activity, activity_labels)
        self.data     = self.all_data[mask]
        self.activity = self.all_activity[mask]
        label_map = {lbl: i for i, lbl in enumerate(sorted(activity_labels))}
        self.targets = torch.LongTensor(
            [label_map[int(a)] for a in self.activity])

    def get_train_val_loader(self, max_samples=None):
        if not self.train:
            raise Exception("get_train_val_loader is only for train mode.")
        indices = list(range(len(self.targets)))
        if max_samples is not None and max_samples < len(indices):
            _tgt = self.targets[indices].numpy()
            indices, _ = train_test_split(indices, train_size=max_samples, stratify=_tgt)
        targets_sub = self.targets[indices].numpy()
        tr_idx, va_idx = train_test_split(
            indices, test_size=self.perc_val, shuffle=True, stratify=targets_sub)
        tr_sampler = BatchSampler(SubsetRandomSampler(tr_idx),
                                  batch_size=self.batch_size, drop_last=True)
        va_sampler = BatchSampler(SubsetRandomSampler(va_idx),
                                  batch_size=len(va_idx), drop_last=False)
        return DataLoader(self, batch_sampler=tr_sampler), \
               DataLoader(self, batch_sampler=va_sampler)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  WISDM  —  Wireless Sensor Data Mining (accelerometer)
#  3 tasks: Walking/Jogging · Upstairs/Downstairs · Sitting/Standing
#  Labels (0-based): Walking=0, Jogging=1, Upstairs=2, Downstairs=3,
#                    Sitting=4, Standing=5
# ══════════════════════════════════════════════════════════════════════════════

_WISDM_PROCESSED  = 'wisdm_processed.npz'
_WISDM_RAW_FILE   = 'WISDM_ar_v1.1_raw.txt'
_WISDM_WINDOW     = 128
_WISDM_STRIDE     = 64
_WISDM_ACTIVITY_MAP = {
    'Walking':    0,
    'Jogging':    1,
    'Upstairs':   2,
    'Downstairs': 3,
    'Sitting':    4,
    'Standing':   5,
}


def _wisdm_preprocess(root):
    """Parse WISDM raw file, segment into windows, split by user, z-score normalise."""
    raw_path = os.path.join(root, _WISDM_RAW_FILE)
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
            act_code = _WISDM_ACTIVITY_MAP.get(activity)
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
        while i + _WISDM_WINDOW <= len(arr):
            window = arr[i:i + _WISDM_WINDOW]
            if uid in train_users:
                X_tr.append(window); y_tr.append(act)
            else:
                X_te.append(window); y_te.append(act)
            i += _WISDM_STRIDE

    X_tr = np.stack(X_tr).astype(np.float32)  # (N, 128, 3)
    y_tr = np.array(y_tr, dtype=np.int64)
    X_te = np.stack(X_te).astype(np.float32)
    y_te = np.array(y_te, dtype=np.int64)

    print(f'[WISDM] train={len(X_tr)}, test={len(X_te)}')
    uniq, counts = np.unique(y_tr, return_counts=True)
    label_names = {v: k for k, v in _WISDM_ACTIVITY_MAP.items()}
    print(f'[WISDM] Activity distribution (train): '
          f'{ {label_names[u]: c for u, c in zip(uniq.tolist(), counts.tolist())} }')

    mean = X_tr.mean(axis=(0, 1))
    std  = X_tr.std(axis=(0, 1)) + 1e-8
    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_te = ((X_te - mean) / std).astype(np.float32)

    out = os.path.join(root, _WISDM_PROCESSED)
    np.savez(out, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
             mean=mean, std=std)
    print(f'[WISDM] Saved -> {out}')


class WISDM_CL(torch.utils.data.Dataset):
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
        processed = os.path.join(root, _WISDM_PROCESSED)

        if not os.path.exists(processed):
            _wisdm_preprocess(root)

        data = np.load(processed)
        if train:
            X, y = data['X_train'], data['y_train']
        else:
            X, y = data['X_test'],  data['y_test']

        self.all_data     = torch.FloatTensor(X)   # (N, 128, 3)
        self.all_activity = y                      # numpy (N,) raw labels 0-5
        self.all_targets  = torch.LongTensor(y)

        self.data     = self.all_data
        self.targets  = self.all_targets
        self.activity = self.all_activity

        self.perc_val   = perc_val
        self.batch_size = batch_size
        self.train      = train

    def choose_subset(self, activity_labels):
        """Select records for the given activity labels and remap to 0..len-1."""
        mask = np.isin(self.all_activity, activity_labels)
        self.data     = self.all_data[mask]
        self.activity = self.all_activity[mask]
        label_map = {lbl: i for i, lbl in enumerate(sorted(activity_labels))}
        self.targets = torch.LongTensor(
            [label_map[int(a)] for a in self.activity])

    def get_train_val_loader(self, max_samples=None):
        if not self.train:
            raise Exception("get_train_val_loader is only for train mode.")
        indices = list(range(len(self.targets)))
        if max_samples is not None and max_samples < len(indices):
            _tgt = self.targets[indices].numpy()
            indices, _ = train_test_split(indices, train_size=max_samples, stratify=_tgt)
        targets_sub = self.targets[indices].numpy()
        tr_idx, va_idx = train_test_split(
            indices, test_size=self.perc_val, shuffle=True, stratify=targets_sub)
        tr_sampler = BatchSampler(SubsetRandomSampler(tr_idx),
                                  batch_size=self.batch_size, drop_last=True)
        va_sampler = BatchSampler(SubsetRandomSampler(va_idx),
                                  batch_size=len(va_idx), drop_last=False)
        return DataLoader(self, batch_sampler=tr_sampler), \
               DataLoader(self, batch_sampler=va_sampler)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]