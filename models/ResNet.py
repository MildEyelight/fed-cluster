import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.fc = nn.Linear(1000, 10)  

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
