import torch

model = torch.hub.load("S-M-J-I/Multimodal-SER",
                       'savee_model', pretrained=True)

print(model)
