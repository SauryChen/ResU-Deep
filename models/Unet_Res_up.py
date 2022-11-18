import torch.nn as nn
import torch
#from torchsummary import summary
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0)

class DoubleResConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleResConv, self).__init__()
        self.conv = nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride = 1),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.ReLU(inplace = True),
            ResidualBlock(out_ch, out_ch, stride = 1),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.ReLU(inplace = True)
        )
    def forward(self, input):
        return self.conv(input)

class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResConv, self).__init__()
        self.conv = nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride = 1),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.ReLU(inplace = True)
        )
    def forward(self, input):
        return self.conv(input)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.xconv = conv1x1(in_channels, out_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = self.xconv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)    # put relu before addition.
        out += residual
        out = self.relu(out)
        return out

class Unet_Res_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Res_up, self).__init__()
        self.conv1 = ResConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ResConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride = 2)

        self.conv3 = ResConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride = 2)

        self.conv4 = ResConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride = 2)

        self.conv5 = ResConv(512, 1024)
        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride = 2)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride = 2)
        self.conv6 = ResConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride = 2)
        self.conv7 = ResConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride = 2)
        self.conv8 = ResConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.conv9 = ResConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        up2 = self.up2(c2)
        up2 = torch.nn.functional.interpolate(up2, size = (c1.size(dim=2), c1.size(dim=3)))
        sum2 = torch.add(c1, up2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        up3 = self.up3(c3)
        up3 = torch.nn.functional.interpolate(up3, size = (c2.size(dim=2), c2.size(dim=3)))
        sum3 = torch.add(c2, up3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        up4 = self.up4(c4)
        up4 = torch.nn.functional.interpolate(up4, size = (c3.size(dim=2), c3.size(dim=3)))
        sum4 = torch.add(c3, up4)

        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up5 = self.up5(c5)
        up5 = torch.nn.functional.interpolate(up5, size = (c4.size(dim=2), c4.size(dim=3)))
        sum5 = torch.add(c4, up5)
        up_6 = self.up6(c5)

        up_6 = torch.nn.functional.interpolate(up_6, size = (c4.size(dim=2), c4.size(dim=3)))
        merge6 = torch.cat([up_6, sum5], dim = 1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        # ! add
        up_7 = torch.nn.functional.interpolate(up_7, size = (c3.size(dim=2), c3.size(dim=3)))
        merge7 = torch.cat([up_7, sum4], dim = 1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        # ! add
        up_8 = torch.nn.functional.interpolate(up_8, size = (c2.size(dim=2), c2.size(dim=3)))
        merge8 = torch.cat([up_8, sum3], dim = 1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        up_9 = torch.nn.functional.interpolate(up_9, size = (c1.size(dim=2), c1.size(dim=3)))
        merge9 = torch.cat([up_9, sum2], dim = 1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        #out = c10
        return out


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = Unet_Res_up(in_ch=33, out_ch=1)
#model = model.to(device)
#summary(model, (33,33,37))
