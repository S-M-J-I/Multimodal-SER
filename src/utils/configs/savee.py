import pandas as pd
import gc
import os
import re


class SAVEEConfigs:
    # A dict that maps the class name to our assigned index (uses: track emotion index for prediction)
    """
    This is for SAVEE ONLY
    """
    class2idx = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "happiness": 3,
        "neutral": 4,
        "sadness": 5,
        "surprise": 6,
    }

    # A dict that maps the index to the class name (uses: decorate prediction)
    idx2class = {v: k for k, v in class2idx.items()}

    # A dict that maps the type given in the file name to our index(uses: dataset preparation)
    tag2idx = {
        "a": 0,
        "d": 1,
        "f": 2,
        "h": 3,
        "n": 4,
        "sa": 5,
        "su": 6,
    }

    # Make a tuple of audio, video, labels
    @staticmethod
    def make_ds_as_list(cls, path):
        audio = []
        video = []
        labels = []
        for dirname, _, filenames in sorted(os.walk(f"{path}Video")):
            for filename in sorted(filenames):
                if filename == "Info.txt":
                    continue

                video_path = os.path.join(dirname, filename)
                label = re.split("(\d+)", filename)[0]
                label = cls.tag2idx[label]
                video.append(video_path)
                labels.append(label)

        for dirname, _, filenames in sorted(os.walk(f"{path}Audio")):
            for filename in sorted(filenames):
                if filename == "Info.txt":
                    continue

                audio_path = os.path.join(dirname, filename)
                audio.append(audio_path)

        return audio, video, labels

    # Create a dataframe
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
