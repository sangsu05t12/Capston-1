import torch
from torch import nn
from torchsummary import summary
import torchvision.models as models

#model = models.resnet18(weights='resnet18')
#model = models.resnet18(pretrained=True)
#model = models.resnet18(weights='imagenet')
#check = torch.load('custom-classifier_resnet_18_final.pth')
#model.load_state_dict(check)

check = torch.load('custom-classifier_resnet_18_final.pth')

print(check)