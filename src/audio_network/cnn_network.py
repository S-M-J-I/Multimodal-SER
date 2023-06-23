from torch import nn


class Conv1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, dilation=2, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
            nn.GroupNorm(1, 128)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
            nn.GroupNorm(1, 128)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
            nn.GroupNorm(1, 128)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out
