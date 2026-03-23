# An audio video-based multi-modal fusion approach for speech emotion recognition

### Status: Accepted at Knowledge-Based Systems (Elsevier), 2026 — In Press, 26 | [Paper Link](https://www.sciencedirect.com/science/article/pii/S0950705126004909)

***Authors: S M Jishanul Islam, Sahid Hossain Mustakim, Musfirat Hossain, Mysun Mashira, Nur Islam Shourav, Md Rayhan Ahmed, Salekul Islam, A.K.M Muzahidul Islam, Swakkhar Shatabda***

## Requirements

**Prerequisite:** A [CUDA-enabled GPU](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf) is preferred. However, for those who would run this code on CPU, ensure to tweak the batch size in correspondence to your hardware capacity. Tweak the batch size in the [hyperparams.py](./src/utils/configs/hyperparams.py) file before running the notebooks. If the installations fail, kindly refer to: [conda instructions](./AlernateInstructions.md).

If frequent problems arise while running on the local environment, kindly resort to the [instructions for cloud notebooks](./cloud_notebooks/CloudInstructions.md), and run on any cloud platform. 

<hr/>

**Step-2:** Clone this repository:
```sh
git clone https://github.com/S-M-J-I/Multimodal-Emotion-Recognition
```

If you have SSH configured:
```sh
git clone git@github.com:S-M-J-I/Multimodal-Emotion-Recognition.git
```

<hr/>

**Step-3:** Install pipenv. Skip if you already have it in your system.
```sh
pip3 install --user pipenv
```
<hr/>

**Step-4:** Install the modules. Run the following command in the terminal:
```sh
pipenv install -r requirements.txt
```

## Run the model
To load the model, you can use `torch.hub()` to load it without having the model's structure:
```python
import torch

# example: load the savee model
model = torch.hub.load("S-M-J-I/Multimodal-SER",'savee_model', pretrained=True, num_classes=7, fine_tune_limit=3)
```
After this, the model will load and it will be ready for use!

Alternatively, this repo contains a [weights directory](./weights/).


## Run the pipelines

To run the notebooks on SAVEE and RAVDESS, we recommend you download the dataset and unpack it in this directory. Then set the path to the directory in their respective notebooks.\
**Note: while setting the file path, ensure the exta '/' is added to the end. Example: `/path_to_dir/`**

To run the model on the datasets, navigate to the individual notebooks made for them in the [explore](./explore/) directory.

Run the following command in the terminal to start the local server:
```sh
pipenv run jupyter notebook
```

### Cite this work
```
@article{ISLAM2026115762,
title = {An audio-video based multi-modal fusion approach for emotion recognition},
journal = {Knowledge-Based Systems},
pages = {115762},
year = {2026},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2026.115762},
url = {https://www.sciencedirect.com/science/article/pii/S0950705126004909},
author = {S M Jishanul Islam and Sahid Hossain Mustakim and Musfirat Hossain and Mysun Mashira and Nur Islam Shourav and Md Rayhan Ahmed and Salekul Islam and A. K M Muzahidul Islam and Swakkhar Shatabda},
keywords = {Deep Learning, Multimodal Deep Learning, Convolutional neural network, Squeeze-and-Excitation network, Frame Filtering, Emotion recognition},
abstract = {Multi-modal deep learning leverages complementary information from multiple data sources, such as audio and video, to improve the robustness of emotion recognition systems. However, effectively integrating heterogeneous features across modalities remains challenging. In this paper, we propose a multi-modal architecture that jointly models audio and visual information for emotion classification. The framework employs a pre-trained R(2+1)D network to extract spatiotemporal representations from video inputs through the combination of 2D spatial and 1D temporal convolutions within a residual learning structure. For the audio modality, multiple acoustic representations—including root-mean-square energy, zero-crossing rate, mel-frequency cepstral coefficients, and mel-spectrogram features averaged along the temporal dimension—are processed using dedicated 1D convolutional networks augmented with a squeeze-and-excitation module for channel-wise feature recalibration. The extracted audio and video features are fused and passed to a Softmax classifier for emotion prediction. The proposed approach is evaluated on three benchmark datasets: the Ryerson Audio–Visual Database of Emotional Speech and Song (RAVDESS), the Surrey Audio–Visual Expressed Emotion (SAVEE), and the Crowd-Sourced Emotional Multimodal Actors Dataset (CREMA-D). Experimental results demonstrate strong performance, achieving F1-scores of 0.9711, 0.9965, and 0.8990 on RAVDESS, SAVEE, and CREMA-D, respectively, showing competitive performance with recent state-of-the-art methods. The code and trained models are publicly available at https://github.com/S-M-J-I/Multimodal-SER.}
}
```

#### For any assistance or issues, kindly open an Issue in this repository.


## Contributions

This repository is not accepting any contributors **OUTSIDE** the author list mentioned. For any issues related to the code, we request you to open an Issue.


