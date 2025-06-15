import torch

class OscillationTracker:
    def __init__(self, threshold: int = 3):
        self.history = []
        self.last_sign = None
        self.flip_count = 0
        self.threshold = threshold

    def update(self, delta):
        current_sign = torch.sign(torch.sum(delta)).item()

        if self.last_sign is not None and current_sign != self.last_sign:
            self.flip_count += 1
        self.last_sign = current_sign
        self.history.append(delta)

    def should_solve(self):
        return self.flip_count >= self.threshold

    def midpoint(self, current_x):
        # Approximate solution: pick midpoint of min/max values seen
        stacked = torch.stack(self.history)
        low = torch.min(stacked, dim=0).values
        high = torch.max(stacked, dim=0).values
        return (low + high) / 2 + current_x  # move towards middle

    def reset(self):
        self.history = []
        self.last_sign = None
        self.flip_count = 0

