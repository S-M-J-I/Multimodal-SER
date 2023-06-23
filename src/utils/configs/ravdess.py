import pandas as pd
import gc
import os


class RAVDESSConfigs:
    """
    This is for RAVDESS ONLY.
    """
    class2idx = {
        "neutral": 0,
        "calm": 1,
        "happy": 2,
        "sad": 3,
        "angry": 4,
        "fearful": 5,
        "disgust": 6,
        "surprised": 7,
    }

    # A dict that maps the index to the class name (uses: decorate prediction)
    idx2class = {v: k for k, v in class2idx.items()}

    # A dict that maps the type given in the file name to our index(uses: dataset preparation)
    tag2idx = {
        "01": 0,
        "02": 1,
        "03": 2,
        "04": 3,
        "05": 4,
        "06": 5,
        "07": 6,
        "08": 7,
    }

    # Make a tuple of audio, video, labels
    @staticmethod
    def make_ds_as_list(cls, path):
        audio = []
        video = []
        labels = []
        for dirname, _, filenames in sorted(os.walk(f"{path}Video")):
            for filename in sorted(filenames):
                video_path = os.path.join(dirname, filename)
                label = filename.split('-')[2]
                label = cls.tag2idx[label]
                video.append(video_path)
                labels.append(label)

        for dirname, _, filenames in sorted(os.walk(f"{path}Audio")):
            for filename in sorted(filenames):
                audio_path = os.path.join(dirname, filename)
                audio.append(audio_path)

        return audio, video, labels

    # Create a dataframe
    @staticmethod
    def make_dataframe(cls, path, augment=0):
        audio, video, labels = cls.make_ds_as_list(path)
        data = pd.DataFrame()
        data['audio_path'] = audio
        data['video_path'] = video
        data['label'] = labels
        data['augment'] = augment

        del audio
        del video
        del labels
        gc.collect()

        return data
