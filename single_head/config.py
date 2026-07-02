"""
Single-Head (DIL) -- shared configuration.

Global constants used across the three entry points (run_tuning.py, run_experiment.py,
run_epoch_analysis.py).

"""
import warnings
warnings.filterwarnings("ignore")
warnings.showwarning = lambda *args, **kwargs: None
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS    = ["mnist", "hhar", "fashion_sw"]
MODELS      = ["gim_alstm_sh", "lstm_sh", "joint_lstm_sh", "lstm_replay_sh", "esn_naive_sh", "esn_ewc_sh", "esn_lwf_sh", "esn_replay_sh","esn_slda_sh", "joint_esn_sh"]
SEED        = 42
RESULTS_DIR = os.path.join(HERE, "..", "results", "single_head")
MNIST_DIR   = os.path.join(HERE, "..", "data", "mnist")
HHAR_DIR      = os.path.join(HERE, "..", "data", "hhar")
FASHION_DIR   = os.path.join(HERE, "..", "data", "fashion_mnist")

SUBSET_MNIST = None
SUBSET_HHAR  = None
SUBSET_FASHION = None

MNIST_DATASET_CONFIGS = {
    "config_1": [[0,1],[2,3],[4,5],[6,7],[8,9]],
    "config_2": [[1,2],[3,4],[5,6],[7,8],[0,9]],
    "config_3": [[0,3],[2,5],[4,7],[6,9],[8,1]],
    "config_4": [[8,3],[2,7],[4,9],[6,1],[0,5]],
    "config_5": [[6,9],[0,7],[8,1],[2,3],[4,5]],
    "config_6": [[8,5],[2,7],[6,3],[0,9],[4,1]],
    "config_7": [[2,5],[4,7],[0,3],[6,9],[8,1]],
    "config_8": [[8,7],[4,5],[0,9],[2,1],[6,3]],
    "config_9": [[0,7],[4,9],[8,1],[6,3],[2,5]],
    "config_10": [[0,3],[6,9],[2,5],[4,1],[8,7]],
}


HHAR_DATASET_CONFIGS = {
    "config_1": [0, 1, 2, 3],
    "config_2": [3, 2, 1, 0],
    "config_3": [0, 2, 1, 3],
    "config_4": [2, 1, 3, 0],
    "config_5": [1, 0, 2, 3],
    "config_6": [3, 1, 2, 0],
    "config_7": [3, 1, 0, 2],
    "config_8": [1, 3, 0, 2],
    "config_9": [1, 2, 0, 3],
    "config_10": [0, 2, 3, 1],
}


FASHION_SW_DATASET_CONFIGS = {
    "config_1": [[0,2],[3,4],[5,9]],
    "config_2": [[5,2],[3,9],[0,4]],
    "config_3": [[3,2],[5,4],[0,9]],
    "config_4": [[0,4],[5,9],[3,2]],
    "config_5": [[0,4],[5,2],[3,9]],
    "config_6": [[3,9],[5,4],[0,2]],
    "config_7": [[0,4],[3,2],[5,9]],
    "config_8": [[3,4],[5,9],[0,2]],
    "config_9": [[3,2],[0,4],[5,9]],
    "config_10": [[5,2],[0,4],[3,9]],
}

DATASET_CONFIGS = {"mnist": MNIST_DATASET_CONFIGS, "hhar": HHAR_DATASET_CONFIGS, "fashion_sw": FASHION_SW_DATASET_CONFIGS}
DEVICE = torch.device("cpu")

EPOCHS = 5  # epochs are fixed at 5 for tuning and training, epoch analysis will run more

TUNE_SUBSET_MNIST = 1200
TUNE_SUBSET_HHAR  = 675
TUNE_SUBSET_FASHION = 1200
TUNE_SPLIT_SEED   = 0      # split tuning data off from training data in tuning


SUBSET_BY_DATASET = {
    "mnist": SUBSET_MNIST, "hhar": SUBSET_HHAR,
    "fashion_sw": SUBSET_FASHION,
}

TUNE_SUBSET_BY_DATASET = {
    "mnist": TUNE_SUBSET_MNIST, "hhar": TUNE_SUBSET_HHAR,
    "fashion_sw": TUNE_SUBSET_FASHION,
}

TUNED_PARAMS_CACHE = os.path.join(HERE, "tuned_params.json")
