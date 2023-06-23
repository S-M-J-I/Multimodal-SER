from torch import nn


class SENet(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.se_net = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        self.global_pooling_bridge = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.global_pooling_bridge(x)
        out = self.flatten(out)
        out = self.se_net(out)
        return out
