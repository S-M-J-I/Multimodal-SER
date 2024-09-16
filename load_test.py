import torch

model = torch.hub.load("S-M-J-I/Multimodal-SER",
                       'savee_model', pretrained=True, num_classes=7, fine_tune_limit=3)

print(model)
