import torch
from torch import nn
from axe_net import AxeNet


class AudioNetwork(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()

        self.zcr_net = AxeNet(reduction=reduction)
        self.rms_net = AxeNet(reduction=reduction)
        self.mfcc_net = AxeNet(reduction=reduction)
        self.mel_net = AxeNet(reduction=reduction)

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x["mfcc"] = x["mfcc"].unsqueeze(dim=1).float()
        x["zcr"] = x["zcr"].unsqueeze(dim=1).float()
        x["mel"] = x["mel"].unsqueeze(dim=1).float()
        x["rms"] = x["rms"].unsqueeze(dim=1).float()

        out_mfcc = self.mfcc_net(x["mfcc"])
        out_mel = self.mel_net(x["mel"])

        out_zcr = self.zcr_net(x["zcr"])
        out_rms = self.rms_net(x["rms"])

        combined = torch.cat([out_mfcc, out_zcr, out_rms, out_mel], dim=1)

        return combined
