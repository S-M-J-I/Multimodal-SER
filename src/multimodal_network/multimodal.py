import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from ..audio_network.audio_net import AudioNetwork
from ..video_network.video_network import VideoNetwork
import gc


__all__ = ['Model', 'model']

models_url = {
    'savee_model': 'https://github.com/S-M-J-I/Multimodal-SER/raw/master/weights/best-multimodal_ravdess.pt',
    'ravdess_model': 'https://github.com/S-M-J-I/Multimodal-SER/raw/master/weights/best-multimodal_savee.pt',
    'crema_model': ''
}


class MainMultimodal(nn.Module):
    def __init__(self, num_classes, fine_tune_limit=3, reduction=16):
        super().__init__()

        self.num_classes = num_classes

#         # define video extractor, cut off FCN layer
        self.video_extractor = VideoNetwork(fine_tune_limit=fine_tune_limit)

        # define audio extractor
        self.audio_extractor = AudioNetwork(reduction=reduction)

        self.fc = nn.Sequential(
            nn.LayerNorm(512 + 5632),
            nn.Dropout(0.5),
            nn.Linear(512 + 5632, num_classes, bias=True),
        )

        # init dual gpu usuage
        self.video_extractor = nn.DataParallel(self.video_extractor)
        self.audio_extractor = nn.DataParallel(self.audio_extractor)

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, video, audio):

        video = torch.permute(video, (0, 2, 1, 3, 4))

        video_feature_values = self.video_extractor(video)

        audio_feature_values = self.audio_extractor(audio)

        del video, audio
        torch.cuda.empty_cache()
        gc.collect()

        combined = torch.cat(
            [video_feature_values, audio_feature_values], dim=1)

        out_logits = self.fc(combined)
        out_softmax = self.softmax(out_logits)

        return out_logits, out_softmax


def model(model_name, pretrained: bool = False, progress: bool = True, **kwargs):
    model = MainMultimodal(**kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(models_url[model_name]))

    return model
