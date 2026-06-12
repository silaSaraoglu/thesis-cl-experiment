import os
import sys
import torch
import torch.nn as nn

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from clrnn.deep_esn import DeepReservoirClassifier


class MultiHeadESN(DeepReservoirClassifier):
    """
    ESN reservoir (frozen) with a ModuleDict of per-task linear heads.

    Before each task:
      model.add_head(task_id)  — register a new Linear head
      model.set_task(task_id)  — route forward() through that head

    At test time the oracle task ID is provided (TIL upper bound).
    """

    def __init__(self, input_size, args, device, num_classes=2):
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

    @property
    def device(self):
        return self._device

    def set_task(self, task_id):
        self._task_id = task_id

    def forward(self, x):
        with torch.no_grad():
            _, states = self.hidden(x)
        features = states[-1]
        return self.heads[str(self._task_id)](features)
