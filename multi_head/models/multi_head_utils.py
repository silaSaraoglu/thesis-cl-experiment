"""
Multi-head wrappers for Task-Incremental Learning (TIL / Setting 2).

Each class subclasses the corresponding repo model, replacing its fixed output
head with a ModuleDict of per-task nn.Linear heads.

Before each task:
  model.add_head(task_id)   — register a new Linear head for this task
  model.set_task(task_id)   — route forward() through task_id's head

At test time the oracle task ID is provided (TIL upper bound).
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from core.BaselineRNN import BaselineRNN
from clrnn.deep_esn import DeepReservoirClassifier

class MultiHeadLSTM(BaselineRNN):
    """
    BaselineRNN (LSTM) with a ModuleDict of per-task linear heads instead of
    a single shared output layer.
    """

    def __init__(self, input_size, hidden_size, device, batch_size, num_classes = 2):
        super().__init__(input_size, hidden_size, output_size=num_classes,
device=device, lstm=True, batch_size=batch_size, num_layers=1)
        
        del self.linear
        self.heads       = nn.ModuleDict()
        self.num_classes = num_classes
        self._task_id    = 0

    def add_head(self, task_id):
        key = str(task_id)
        if key not in self.heads:
            self.heads[key] = nn.Linear(self.hidden_size, self.num_classes).to(self.device)

    def set_task(self, task_id):
        self._task_id = task_id

    def forward(self, x, hidden_state):
        out, h_new = self.rnn_module(x, hidden_state)
        features   = out[:, -1, :]
        logits     = self.heads[str(self._task_id)](features)
        return h_new, logits.unsqueeze(1)   # (h_new, (B, 1, num_classes))

class MultiHeadESN(DeepReservoirClassifier):
    """
    ESN reservoir (frozen) with a ModuleDict of per-task linear heads.
    """

    def __init__(self, input_size, args, device, num_classes = 2):
        super().__init__(
            input_size=input_size,
            num_classes=num_classes,
            units=args.esn_units,
            layers=1,
            concat=True,
            feedforward_layers=1,
            spectral_radius=0.99,
            leaky=1.0,
            input_scaling=1.0,
            return_sequences=False,
        )

        for p in self.hidden.parameters():
            p.requires_grad_(False)

        del self.output
        self.feature_size = self.hidden.layers_units
        self.heads        = nn.ModuleDict()
        self.num_classes  = num_classes
        self._device      = device
        self._task_id     = 0
        self.to(device)

    def add_head(self, task_id):
        key = str(task_id)
        if key not in self.heads:
            self.heads[key] = nn.Linear(self.feature_size, self.num_classes).to(self._device)

    def set_task(self, task_id):
        self._task_id = task_id

    def forward(self, x):
        with torch.no_grad():
            _, states = self.hidden(x)
        features = states[-1]
        return self.heads[str(self._task_id)](features)

# ── shared helper ─────────────────────────────────────────────────────────────

def _forward_mh(model, x):
    """Single inference pass for a multi-head LSTM; returns logits (B, num_classes)."""
    hidden_state = model.reset_memory_state(batch_size=x.size(0))
    _, out = model(x, hidden_state)    # out: (B, 1, num_classes)
    return out[:, -1, :]

def _predict_mh(model, loader):
    """Collect predictions and labels over a DataLoader for a MultiHeadESN."""
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(1).numpy()
            all_p.extend(preds); all_l.extend(y.numpy())
    return np.array(all_p), np.array(all_l)
