from torch import nn
from cnn_network import Conv1D
from se_net import SENet


class AxeNet(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()

        self.cnn_encoder = Conv1D()

        self.se_net = SENet(channels=128, reduction=reduction)

        self.flatten = nn.Flatten()

        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.cnn_encoder(x)
        residual = out

        attn_out = self.se_net(out)

        attn_out = attn_out.unsqueeze(dim=-1)

        out_total = attn_out * residual
        out_total = self.flatten(out_total)

        return out_total
