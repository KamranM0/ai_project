import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=False)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
