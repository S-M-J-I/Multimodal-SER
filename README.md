# Multimodal Emotion Recognition (Audio-Visual Fusion)

<p align="center">
  <b>Official Implementation of our Knowledge-Based Systems (Elsevier) 2026 Paper</b><br>
  <i>An audio-video based multi-modal fusion approach for emotion recognition</i><br>
</p>

<p align="center">
  📄 <a href="https://www.sciencedirect.com/science/article/pii/S0950705126004909">Paper</a> • 
  💻 <a href="https://github.com/S-M-J-I/Multimodal-SER">Code</a> • 
  🧠 <b>PyTorch</b> • 🎯 <b>Multimodal Learning</b>
</p>

**Authors: S M Jishanul Islam, Sahid Hossain Mustakim, Musfirat Hossain, Mysun Mashira, Nur Islam Shourav, Md Rayhan Ahmed, Salekul Islam, A.K.M Muzahidul Islam, Swakkhar Shatabda**

Keywords: Multimodal Emotion Recognition, Speech Emotion Recognition, Audio-Visual Learning, Deep Learning, R(2+1)D, Squeeze-and-Excitation, CNN, PyTorch, Affective Computing, Human Behavior Analysis

---

## 🔥 Highlights

- 🚀 **Accepted at Knowledge-Based Systems (Elsevier), 2026 (In Press)**
- 🎧 + 🎥 **Audio-Visual Multimodal Fusion Framework**
- 🧠 Combines **CNN, SE Networks, and R(2+1)D**
- 📊 Achieves **state-of-the-art performance** on:
  - RAVDESS
  - SAVEE
  - CREMA-D
- ⚡ **Pretrained models available via `torch.hub`**

---

## 🧩 Overview

Emotion recognition from speech is inherently **multimodal**, involving both **acoustic signals** and **facial expressions**.

This repository presents a **deep multimodal architecture** that:

- Extracts **spatiotemporal features** from video using **R(2+1)D**
- Processes **acoustic features** using **1D CNN + Squeeze-and-Excitation (SE)**
- Learns a **joint embedding space** via fusion
- Performs **robust emotion classification**

---

## 🏗️ Architecture

```
Video → R(2+1)D Encoder ┐
                        ├──► Fusion ──► Classifier (Softmax)
Audio → 1D CNN + SE ────┘
```

### Key Components
- **Video Encoder**
  - Pretrained R(2+1)D
  - Captures spatial + temporal dependencies

- **Audio Encoder**
  - Features:
    - MFCC
    - Mel-Spectrogram
    - RMS Energy
    - Zero Crossing Rate
  - Enhanced with **Squeeze-and-Excitation (SE)**

- **Fusion**
  - Multimodal feature aggregation

---

## 📊 Results

| Dataset  | F1-Score |
|----------|--------|
| RAVDESS  | **0.9711** |
| SAVEE    | **0.9965** |
| CREMA-D  | **0.8990** |

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/S-M-J-I/Multimodal-Emotion-Recognition
cd Multimodal-Emotion-Recognition
```

2. Install Dependencies
```bash
pip3 install --user pipenv
pipenv install -r requirements.txt
```

## 🚀 Quick Start (Pretrained Model)

Load model directly using PyTorch Hub:
```py
import torch

model = torch.hub.load(
    "S-M-J-I/Multimodal-SER",
    "savee_model",
    pretrained=True,
    num_classes=7,
    fine_tune_limit=3
)
```
✔ No need to define architecture manually

## 📂 Dataset Setup

Download datasets:
1. RAVDESS
2. SAVEE
3. CREMA-D

Place them locally and update paths inside notebooks.

> ⚠️ Important:
> /path_to_dataset/
> (Ensure trailing /)

## 🧪 Running Experiments
```bash
pipenv run jupyter notebook
```

Navigate to `explore/`. Each notebook includes:
1. Data preprocessing
2. Feature extraction
3. Training & evaluation

## 🧠 Technical Contributions
✅ Multimodal fusion of audio + video

✅ Use of R(2+1)D for emotion-aware video encoding

✅ Channel recalibration via Squeeze-and-Excitation

✅ Robust performance across multiple datasets

✅ Lightweight and reproducible pipeline

If you use this work, please cite:

```
@article{ISLAM2026115762,
  title = {An audio-video based multi-modal fusion approach for emotion recognition},
  journal = {Knowledge-Based Systems},
  pages = {115762},
  year = {2026},
  issn = {0950-7051},
  doi = {https://doi.org/10.1016/j.knosys.2026.115762},
  url = {https://www.sciencedirect.com/science/article/pii/S0950705126004909},
  author = {S M Jishanul Islam and Sahid Hossain Mustakim and Musfirat Hossain and Mysun Mashira and Nur Islam Shourav and Md Rayhan Ahmed and Salekul Islam and A. K M Muzahidul Islam and Swakkhar Shatabda}, keywords = {Deep Learning, Multimodal Deep Learning, Convolutional neural network, Squeeze-and-Excitation network, Frame Filtering, Emotion recognition},
  abstract = {Multi-modal deep learning leverages complementary information from multiple data sources, such as audio and video, to improve the robustness of emotion recognition systems. However, effectively integrating heterogeneous features across modalities remains challenging. In this paper, we propose a multi-modal architecture that jointly models audio and visual information for emotion classification. The framework employs a pre-trained R(2+1)D network to extract spatiotemporal representations from video inputs through the combination of 2D spatial and 1D temporal convolutions within a residual learning structure. For the audio modality, multiple acoustic representations—including root-mean-square energy, zero-crossing rate, mel-frequency cepstral coefficients, and mel-spectrogram features averaged along the temporal dimension—are processed using dedicated 1D convolutional networks augmented with a squeeze-and-excitation module for channel-wise feature recalibration. The extracted audio and video features are fused and passed to a Softmax classifier for emotion prediction. The proposed approach is evaluated on three benchmark datasets: the Ryerson Audio–Visual Database of Emotional Speech and Song (RAVDESS), the Surrey Audio–Visual Expressed Emotion (SAVEE), and the Crowd-Sourced Emotional Multimodal Actors Dataset (CREMA-D). Experimental results demonstrate strong performance, achieving F1-scores of 0.9711, 0.9965, and 0.8990 on RAVDESS, SAVEE, and CREMA-D, respectively, showing competitive performance with recent state-of-the-art methods. The code and trained models are publicly available at https://github.com/S-M-J-I/Multimodal-SER.}
}
```

## ⭐ Support & Community

If you find this repository useful:
⭐ Star this repo
🍴 Fork for your research
🧾 Cite our paper
🛠️ Issues

## 🔒 Contribution Policy

This repository is not open for external code contributions.


## 🌐 Links
📄 Paper: https://www.sciencedirect.com/science/article/pii/S0950705126004909

💻 Code: https://github.com/S-M-J-I/Multimodal-SER

<p align="center"> <b>Built for research. Designed for reproducibility.</b> </p>
