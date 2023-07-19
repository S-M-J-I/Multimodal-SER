import librosa
import numpy as np
import torch
"""
    Defining the helper functions for the Audio MFCC technique
"""

# audio effects


def audio_effects(audio, sample_rate, augment=1):
    data = None
    if augment == 1:
        data = librosa.effects.percussive(y=audio)
    elif augment == 2:
        data = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=3)
    return data


# normalize the audio wave
def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


def feature_extractor(file, augment=0, test=False):

    attempt = 0
    while True:
        try:
            data, sample_rate = librosa.load(file)
            break
        except:
            if attempt == 50:
                print("failed trying to find audio file", file)
                break
            print("Audio file not read. Trying again")
            attempt += 1

    if augment > 0:
        data = audio_effects(data, sample_rate, augment=augment)

    data = normalize_audio(data)

    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=data)[0]
    zcr /= zcr.max()
    zcr = zcr[0:(0+128)]
    if len(zcr) < 128:
        zcr = librosa.util.fix_length(zcr, size=128)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(
        y=data, sr=sample_rate, n_mfcc=128).T, axis=0)
    mfcc /= mfcc.max()

    # Root Mean Square Value
    rms = librosa.feature.rms(y=data)[0]
    rms /= rms.max()
    rms = rms[0:(0+128)]
    if len(rms) < 128:
        rms = librosa.util.fix_length(rms, size=128)
#     result = np.vstack((result, rms))

    # MelSpectogram
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    mel = librosa.amplitude_to_db(mel, ref=np.max)
    mel = np.mean(mel.T, axis=0)
    mel /= mel.sum()

    if test:
        return_dict = {
            "raw": data,
            "sr": sample_rate,
            "zcr": zcr,
            "mfcc": mfcc,
            "rms": rms,
            "mel": mel
        }
    else:
        return_dict = {
            "zcr": zcr,
            "mfcc": mfcc,
            "rms": rms,
            "mel": mel
        }
    return return_dict


def dict_to_tensor(dictionary, device):
    out_dict = {}
    for item in dictionary.items():
        if item[0] == "sr":
            out_dict[item[0]] = torch.tensor(item[1]).to(device=device)
        else:
            out_dict[item[0]] = torch.from_numpy(item[1]).to(device=device)

    return out_dict
