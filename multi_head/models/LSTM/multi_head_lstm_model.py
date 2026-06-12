import os
import sys
import torch
import torch.nn as nn

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from core.BaselineRNN import BaselineRNN


class MultiHeadLSTM(BaselineRNN):
    """
    BaselineRNN (LSTM) with a ModuleDict of per-task linear heads instead of
    a single shared output layer.

    Before each task:
      model.add_head(task_id)  — register a new Linear head
      model.set_task(task_id)  — route forward() through that head

    At test time the oracle task ID is provided (TIL upper bound).
    """

    def __init__(self, input_size, hidden_size, device, batch_size, num_classes=2):
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

    def forward(self, x):
        hidden_state = self.reset_memory_state(batch_size=x.size(0))
        out, _       = self.rnn_module(x, hidden_state)
        features     = out[:, -1, :]
        return self.heads[str(self._task_id)](features)
