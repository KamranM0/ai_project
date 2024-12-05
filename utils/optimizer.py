# utils/optimizer.py
import torch.optim as optim

def get_optimizer(model, optimizer_name, lr=1e-3):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=5e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
