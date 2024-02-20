# -*- coding: utf-8 -*-
"""
Implementation of ResNet
Paper: https://arxiv.org/pdf/1512.03385.pdf
REF:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py
https://github.com/pytorch/vision/blob/e6b4078ec73c2cf4fd4432e19c782db58719fb99/torchvision/models/resnet.py
"""

import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None):
        super(block, self).__init__()
        # resnet is a succession of similar blocks with different number of repetitions and sizes
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
       
        self.identity_downsample = identity_downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight)
                try:
                    m.bias.data.fill_(0.01)
                except:
                    pass
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
    
        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        out += identity
        out = self.relu(out)

        return out

        
class resNet(nn.Module):
    def __init__(self, in_channels, layers_mult):
        super(resNet, self).__init__()
        self.in_channels = 64
        
        # create layers
        # input: 224x224 image
        #P= np.ceil(((out -1)*S -W +1 + (K-1))/2)
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers_mult[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers_mult[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers_mult[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers_mult[3], stride=2)
    
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)
        #self.out = nn.Softmax(dim=1)       
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        #out = self.out(out)

        return out


    def _make_layer(self, block, out_channels, multiplier, stride):
        # multiplies block layers
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels))
            
        layers += [block(self.in_channels, out_channels, stride, identity_downsample)]  # 1st layer, downsample
        self.in_channels = out_channels 

        for m in range(multiplier-1):
            layers += [block(self.in_channels, out_channels, stride=1)]

        return nn.Sequential(*layers)
    
def test():
    layers_mult = [2, 2, 2, 2]  # 18 layer
    layers_mult = [3, 4, 6, 3]  # 34 layer
    net = resNet(1, layers_mult)
    y = net(torch.randn(4, 1, 224)) # [batch, channels, timesteps]
    print(y.size())
    print(y)


#test()
