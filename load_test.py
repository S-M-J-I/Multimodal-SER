import torch
import os

LOCAL_PATH = os.getcwd()
GITHUB_PATH = "https://github.com/S-M-J-I/Multimodal-SER"

print("Loading from:", LOCAL_PATH)
model = torch.hub.load(LOCAL_PATH,
                       'savee_model', pretrained=True, num_classes=7, fine_tune_limit=3, source='local')

print(model)
