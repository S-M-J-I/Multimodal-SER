from torch import nn
from torchvision.models.video import r2plus1d_18
from passthrough import PassThrough


class VideoNetwork(nn.Module):
    def __init__(self, fine_tune_limit=3):
        super().__init__()

        R2plus1D = r2plus1d_18(weights='KINETICS400_V1')

        self.net = R2plus1D
        self.net.fc = PassThrough()
        count = 0  # keep track for layer count
        # get the length of layers
        length = sum(1 for _ in self.net.children())
        # set the limit [if length is 7, then limit = 7-2(default) = 5 ---> if count is = or above this we set to trainable]
        limit = length - fine_tune_limit

        for child in self.net.children():
            if count >= limit:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

            count += 1

    def forward(self, x):
        out = self.net(x)
        return out
