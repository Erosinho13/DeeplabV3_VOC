import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class ASPP(nn.Module):
    
    def __init__(self, dilation, in_channels=512, hidden_channels=256, size=14):
        
        super().__init__()
        
        self.aspp = []
        
        for d in dilation:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=3,
                              padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(hidden_channels, momentum=0.9997),
                    nn.ReLU()
                )
            )
        
        self.aspp = nn.ModuleList(self.aspp)

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(size),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, momentum=0.9997),
            nn.ReLU()
        )
    
    def forward(self, x):
        
        aspp = [l(x) for l in self.aspp] + [self.pooling(x)]
        res = torch.cat(aspp, dim=1)
        
        return res
        
    
class DeepLabV3(nn.Module):
    
    def __init__(self, dilation, num_classes):
        super().__init__()
        
        resnet18 = models.resnet18(pretrained=True)
        
        for k in resnet18._modules:
        if k == 'layer4':
            break
        print(resnet18._modules[k])
        
        ...
        
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=1, dilation=2, bias=False),
#             nn.BatchNorm2d(512, momentum=0.9997),
#             nn.ReLU()
#         )
        
#         self.aspp = ASPP(dilation, in_channels=512, hidden_channels=256, size=14)
    
#         self.layer5 = nn.Sequential(
#             nn.Conv2d((len(dilation)+1)*256, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256, momentum=0.9997),
#             nn.ReLU()
#         )
    
#         self.out_layer = nn.Sequential(
#             nn.Conv2d(256, num_classes, kernel_size=1),
#             nn.UpsamplingBilinear2d(scale_factor=16)
#         )

    def forward(self, x):
        
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.aspp(x)
#         x = self.layer5(x)
#         x = self.out_layer(x)
        # ups = F.interpolate(x, size=224, mode='bilinear', align_corners=True)

        return x