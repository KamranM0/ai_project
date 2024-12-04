import torch
import torch.nn as nn
import torchvision.models as model

class ExModel(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        self.resnet18 = model.resnet18(pretrained=False)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))    
        
        self.classifier = torch.nn.Linear(512, 15) #TODO Change this classifier according to your application!


    def forward(self, image):
        # Get predictions from ResNet18
        resnet_pred = self.resnet18(image).squeeze()
        out = self.classifier(resnet_pred)
        
        return out
    
def get_resnet18(num_classes, transfer_learning=True):
    model_ = model.resnet18(pretrained=transfer_learning)
    if transfer_learning:
        for param in model_.parameters():
            param.requires_grad = False
    model_.fc = nn.Linear(model_.fc.in_features, num_classes)
    return model_