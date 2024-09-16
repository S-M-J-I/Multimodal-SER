import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from IPython.display import Video
from ..configs.ravdess import RAVDESSConfigs
import numpy as np
"""
    Defining the helper functions
"""

# Get the melspec of the audio as image/np 2d array


def wav2melSpec(AUDIO_PATH):
    audio, sr = librosa.load(AUDIO_PATH)
    return librosa.feature.melspectrogram(y=audio, sr=sr)


# Show the image spectogram
def imgSpec(ms_feature):
    fig, ax = plt.subplots()
    ms_dB = librosa.power_to_db(ms_feature, ref=np.max)
    print(ms_feature.shape)
    img = librosa.display.specshow(ms_dB, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

# Hear the audio


def hear_audio(AUDIO_PATH):
    audio, sr = librosa.load(AUDIO_PATH)

    print("\t", end="")
    ipd.display(ipd.Audio(data=audio, rate=sr))

# Show 1 example


def show_example(video_path, audio_path, prediction=None, actual=None, save_memory=False, idx2class=RAVDESSConfigs().idx2class):
    if prediction is not None:
        print("Predicted Label:", idx2class[prediction])
    print("Actual Label:", idx2class[actual])

    if save_memory is False:
        print("Video path:", video_path)
        ipd.display(Video(video_path, embed=True, width=400, height=300))

        # display(show_video(video_path))
        print("Audio path:", audio_path)
        hear_audio(audio_path)
