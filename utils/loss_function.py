# utils/loss_functions.py
from torch.nn import functional as F
import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probabilities
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
