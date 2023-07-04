import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(33, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 1, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 1, stride=2)
        self.layer4 = self.make_layer(ResBlock, 128, 1, stride=2)

        self.outlayer = nn.Conv2d(128,1, kernel_size=1, stride=1, padding=0, bias = False)

        #self.fc = nn.Linear(512, num_classes) 
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.interpolate(out, size = (x.size(dim=2), x.size(dim=3)))
        out = self.outlayer(out)
        out = nn.Sigmoid()(out)
        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        #out = self.fc(out)
        return out
'''
model = ResNet().to(device)
summary(model, (33, 121, 281))
'''
