import torch

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
