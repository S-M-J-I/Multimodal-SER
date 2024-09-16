import torch
import random
from torch.utils.data import Dataset
from ..transforms.video_transforms import *
from ..transforms.audio_transforms import *
from ..helpers.functions import *
from torchvision.io import read_image, read_video
import gc

random.seed(42)

# The dataset to use in the dataloader


class Dataset(Dataset):
    def __init__(
        self,
        dataframe,
        video_frame_transform=None,
        video_strategy='optimal',
        cut_video=False,
        cut_audio=False,
        selection="quartile",
        variant="all"
    ):

        self.cut_video = cut_video
        self.cut_audio = cut_audio

        self.examples = dataframe
        self.video_frame_transform = {
            0: video_frame_transform,
            1: video_frame_augment_color,
            2: video_frame_augment_persp
        }

        self.video_strategy = video_strategy
        self.selection = selection
        self.variant = variant

        del dataframe, video_frame_transform

    def __len__(self):
        return len(self.examples)

    def __optimal_strategy(self, video, augment=0):

        frames = []

        q1_point = video.shape[0] // 4
        q2_point = video.shape[0] // 2
        q3_point = int((video.shape[0] * (3/4)))
        length = video.shape[0]

        q1_q2_mid = int((q1_point + (q2_point - 5))//2)
        q2_q3_mid = int((q2_point + (q3_point - 5))//2)

        q1_lb = q1_point-10
        q1_up = q1_q2_mid

        q2_lb = q2_point-10
        q2_up = q2_q3_mid

        q3_lb = q3_point-10

        six_frame_lower_bound = (
            self.selection == "six" and self.variant == "lower")
        six_frame_upper_bound = (
            self.selection == "six" and self.variant == "upper")
        original_strategy = (
            self.selection == "quartile" and self.variant == "all")

        # select random starting frames
        if six_frame_lower_bound or original_strategy:
            frames.append(self.video_frame_transform[augment](video[q1_lb]))
        frames.append(self.video_frame_transform[augment](video[q1_point]))
        if six_frame_upper_bound or original_strategy:
            frames.append(self.video_frame_transform[augment](video[q1_up]))

        # select random mid frames
        if six_frame_lower_bound or original_strategy:
            frames.append(self.video_frame_transform[augment](video[q2_lb]))
        frames.append(self.video_frame_transform[augment](video[q2_point]))
        if six_frame_upper_bound or original_strategy:
            frames.append(self.video_frame_transform[augment](video[q2_up]))

        # select random end frames
        if six_frame_lower_bound or original_strategy:
            frames.append(self.video_frame_transform[augment](video[q3_lb]))
        frames.append(self.video_frame_transform[augment](video[q3_point]))
        if six_frame_upper_bound or original_strategy:
            frames.append(self.video_frame_transform[augment](video[-1]))

        frames = torch.stack(frames)

        return frames

    def __audio_extraction(self, audio, augment=0):
        feats = feature_extractor(audio, augment)
        return dict_to_tensor(feats)

    def __getitem__(self, idx):
        df = self.examples.iloc[idx]
        audio_path, video_path, label, augment = df["audio_path"], df["video_path"], df["label"], df['augment']

        if self.cut_video == False:

            attempt = 0
            while True:
                try:
                    video = read_video(video_path, pts_unit='sec')[0]
                    break
                except:
                    if attempt == 50:
                        print("Video file can't be read", video_path)
                        break
                    print("Video file not read. Trying again")
                    attempt += 1

            video = torch.permute(video, (0, 3, 1, 2))

            if self.video_strategy == 'optimal':
                video = self.__optimal_strategy(video, augment)
            elif self.video_strategy == 'all':
                video = self.__all_strategy(video)
        else:
            video = torch.zeros((1, 1))

        if self.cut_audio == False:
            audio = self.__audio_extraction(audio_path, augment)
        else:
            audio = torch.zeros((1, 1))

        del df
        gc.collect()

        return video, audio, label, video_path, audio_path
