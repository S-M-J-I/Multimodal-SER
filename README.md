# An audio video-based multi-modal fusion approach for speech emotion recognition 

### Status: Submitted for Review

***Authors: S M JISHANUL ISLAM, SAHID HOSSAIN MUSTAKIM, MUSFIRAT HOSSAIN, MYSUN
MASHIRA, NUR ISLAM SHOURAV, MD. RAYHAN AHMED, SALEKUL ISLAM, A.K.M. MUZAHIDUL ISLAM, AND SWAKKHAR SHATABDA***

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


#### For any assistance or issues, kindly open an Issue in this repository.


## Contributions

This repository is not accepting any contributors **OUTSIDE** the author list mentioned. For any issues related to the code, we request you to open an Issue.