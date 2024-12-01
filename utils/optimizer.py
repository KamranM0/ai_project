# utils/optimizer.py
import torch.optim as optim

def get_optimizer(model, optimizer_name, lr=1e-3):
    """
    Returns the optimizer for the given model and optimizer name.
    
    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        optimizer_name (str): The name of the optimizer (e.g., "adam", "sgd").
        lr (float): The learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
