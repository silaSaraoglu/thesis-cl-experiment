import torch
class BalancedReservoirBuffer:
    def __init__(self, size):
        self.size = size
        self.groups = []

    def set_per_task_length(self, size, num_groups):
        base = size // num_groups
        lengths = [base] * num_groups
        for i in range(size - base * num_groups):
            lengths[i] += 1
        return lengths

    def update(self, X, Y, task_id=None):
        if task_id is None:
            task_id = len(self.groups)
        w = torch.rand(X.size(0))
        order = torch.argsort(w, descending=True)
        self.groups.append({"X": X[order], "Y": Y[order], "w": w[order], "t": int(task_id)})
        lengths = self.set_per_task_length(self.size, len(self.groups))
        for ll, g in zip(lengths, self.groups):
            if g["X"].size(0) > ll:
                g["X"], g["Y"], g["w"] = g["X"][:ll], g["Y"][:ll], g["w"][:ll]

    def data(self):
        if not self.groups:
            return None, None
        return (torch.cat([g["X"] for g in self.groups]),
                torch.cat([g["Y"] for g in self.groups]))

    def data_with_tasks(self):
        if not self.groups:
            return None, None, None
        X = torch.cat([g["X"] for g in self.groups])
        Y = torch.cat([g["Y"] for g in self.groups])
        T = torch.cat([torch.full((g["X"].size(0),), g["t"], dtype=torch.long)
                       for g in self.groups])
        return X, Y, T
