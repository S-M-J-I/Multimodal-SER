# Steps for using conda

Create a conda virtual environment and activate it.
```sh
conda create -n multimodal-ser
conda activate multimodal-ser
```

Setup conda environment for GPU usage:
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

For CPU-only users:
```sh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Clone this repository:
```sh
git clone https://github.com/S-M-J-I/Multimodal-SER
```

And run:
```sh
pip3 install -r requirements.txt
```



