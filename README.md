# An audio video-based multi-modal fusion approach for speech emotion recognition 

### Status: Submitted to IEEE Access
Link: 

***Authors: S M JISHANUL ISLAM, SAHID HOSSAIN MUSTAKIM, MUSFIRAT HOSSAIN, MYSUN
MASHIRA, NUR ISLAM SHOURAV, MD. RAYHAN AHMED, SALEKUL ISLAM, SWAKKHAR SHATABDA, AND A.K.M. MUZAHIDUL ISLAM***

## Requirements

**Step-1:** A [CUDA-enabled GPU](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf) is preferred. However, for those who would run this code on CPU, ensure to tweak the batch size in correspondence to your hardware capacity. Tweak the batch size in this [file](./src/utils/configs/hyperparams.py) before running the notebooks.

**Step-2:** Clone this repository:
```sh
git clone https://github.com/S-M-J-I/Multimodal-Emotion-Recognition
```

If you have SSH configured:
```sh
git clone git@github.com:S-M-J-I/Multimodal-Emotion-Recognition.git
```

**Step-3:** Run in terminal:
```sh
pip install -r requirements.txt
```


## Run the pipelines

To run the notebooks on SAVEE and RAVDESS, we recommend you download the dataset and unpack it in this directory. Then set the path for in their respective notebooks.\
**Note: while setting the file path, ensure the exta '/' is added to the end. Example: `/path_to_dir/`**

To run the model on the datasets, navigate to the individual notebooks made for them in the [explore](./explore/) directory.

Run the following command in the terminal to start the local server:
```sh
pipenv run jupyter notebook
```

## Weights
To obtain the weights of the model, kindly access the following [link](https://drive.google.com/drive/folders/141iQxVmjnL0zsWhjmakI6stK2CdoeRln?usp=sharing)

For any assistance or issues, kindly open an Issue in this repository.