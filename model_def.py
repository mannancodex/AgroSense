import torch.nn as nn

class GrowthNet(nn.Module):
    """Four-layer feed-forward net (<10k parameters) -> 0-100 growth health score."""
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x).squeeze(-1) * 100
