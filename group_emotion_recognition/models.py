import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pickle
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)    

    def forward(self, x):
        #y_pred = torch.sigmoid(self.linear(x))
        y_pred = self.linear(x)
        return y_pred

    def compute_l2_loss(self, w):
        return torch.square(w).sum() 

# Define model
class DNN(nn.Module):
    def __init__(self, indim):
        super(DNN, self).__init__()
        self.ln1 = nn.Linear(indim, 64)
        self.dp = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(64)        
        self.ReLU = nn.LeakyReLU()

        self.ln2 = nn.Linear(64, 128)
        self.dp = nn.Dropout(p=0.4)
        self.bn2 = nn.BatchNorm1d(128)        
        self.ReLU = nn.LeakyReLU()

        self.ln3 = nn.Linear(128, 256)
        self.dp = nn.Dropout(p=0.4)        
        self.bn3 = nn.BatchNorm1d(256)
        self.ReLU = nn.LeakyReLU()

        self.ln4 = nn.Linear(256, 512)
        self.dp = nn.Dropout(p=0.4)                
        self.bn4 = nn.BatchNorm1d(512)        
        self.ReLU = nn.LeakyReLU()

        self.ln5 = nn.Linear(512, 2)
        self.out = nn.Softmax(dim=1)       
        

    def forward(self, x):
        x = self.ln1(x)  # >272
        x = self.dp(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.ln2(x)  # 272>544
        x = self.dp(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        
        x = self.ln3(x)  # 544 > 272
        x = self.dp(x)
        x = self.bn3(x)
        x = self.ReLU(x)

        x = self.ln4(x)  # 544 > 272
        x = self.dp(x)
        x = self.bn4(x)
        x = self.ReLU(x)

        x = self.ln5(x)  # 272 > 2

        pred = self.out(x)
        return pred


class FCN(nn.Module):
    def __init__(self, indim):
        super(FCN, self).__init__()
        self.ln1 = nn.Linear(indim, 128)
        self.dp = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(128)        
        self.ReLU = nn.ReLU()

        self.ln2 = nn.Linear(128, 256)
        self.dp = nn.Dropout(p=0.2)        
        self.bn2 = nn.BatchNorm1d(256)
        self.ReLU = nn.ReLU()

        self.ln3 = nn.Linear(256, 128)
        self.dp = nn.Dropout(p=0.3)
        self.bn3 = nn.BatchNorm1d(128)        
        self.ReLU = nn.ReLU()

        self.ln4 = nn.Linear(128, 1)
        #self.out = nn.Softmax(dim=1)       
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.ln1(x)  
        x = self.dp(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.ln2(x)  
        x = self.dp(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        
        x = self.ln3(x)  
        x = self.dp(x)
        x = self.bn3(x)
        x = self.ReLU(x)

        x = self.ln4(x)  
        #pred = self.out(x)
        return x


class simpleDNN(nn.Module):
    def __init__(self, indim):
        super(simpleDNN, self).__init__()
        self.ln1 = nn.Linear(indim, 128)  # 128 256
        self.dp = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(128)        
        self.ReLU = nn.LeakyReLU()

        self.ln2 = nn.Linear(128, 256)
        self.dp = nn.Dropout(p=0.4)        
        self.bn2 = nn.BatchNorm1d(256)
        self.ReLU = nn.LeakyReLU()

        self.ln4 = nn.Linear(256, 2)
        self.out = nn.Softmax(dim=1)       
        

    def forward(self, x):
        x = self.ln1(x)  
        x = self.dp(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.ln2(x)  
        x = self.dp(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        
        x = self.ln4(x)
        
        x = self.out(x)
        return x



class FCN_Cv(nn.Module):
    # https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
    def __init__(self, ch):
        super(FCN_Cv, self).__init__()
        # (W-K +2*P)/S+1
        self.ln1 = nn.Conv1d(ch, 128, kernel_size=8, stride=1, padding=0)
        self.dp = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(128)        
        self.ReLU = nn.LeakyReLU()

        self.ln2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=0)
        self.dp = nn.Dropout(p=0.4)        
        self.bn2 = nn.BatchNorm1d(256)
        self.ReLU = nn.LeakyReLU()

        self.ln3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=0)
        self.dp = nn.Dropout(p=0.4)
        self.bn3 = nn.BatchNorm1d(128)        
        self.ReLU = nn.LeakyReLU()
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.ln4 = nn.Linear(128, 1)
        
        #self.out = nn.Softmax(dim=1)       
        

    def forward(self, x):
        x = self.ln1(x)  
        x = self.dp(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.ln2(x)  
        x = self.dp(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        
        x = self.ln3(x)  
        x = self.dp(x)
        x = self.bn3(x)
        x = self.ReLU(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)        
        x = self.ln4(x)  

        #x = self.out(x)
        return x

class FCN_Cv_2d(nn.Module):
    # https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
    def __init__(self, ch):
        super(FCN_Cv_2d, self).__init__()
        # (W-K +2*P)/S+1
        self.ln1 = nn.Conv2d(ch, 128, kernel_size=8, stride=1, padding=0)
        self.dp = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm2d(128)        
        self.ReLU = nn.LeakyReLU()

        self.ln2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=0)
        self.dp = nn.Dropout(p=0.4)        
        self.bn2 = nn.BatchNorm2d(256)
        self.ReLU = nn.LeakyReLU()

        self.ln3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.dp = nn.Dropout(p=0.4)
        self.bn3 = nn.BatchNorm2d(128)        
        self.ReLU = nn.LeakyReLU()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ln4 = nn.Linear(128, 1)
        
        #self.out = nn.Softmax(dim=1)       
        

    def forward(self, x):
        x = self.ln1(x)  
        x = self.dp(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.ln2(x)  
        x = self.dp(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        
        x = self.ln3(x)  
        x = self.dp(x)
        x = self.bn3(x)
        x = self.ReLU(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)        
        x = self.ln4(x)  

        #x = self.out(x)
        return x


class MLP(nn.Module):
    # https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
    
    def __init__(self, indim):
        super(MLP, self).__init__()
        
        self.dp1 = nn.Dropout(p=0.1)
        self.ln1 = nn.Linear(indim, 500)
        self.ReLU = nn.ReLU()

        self.dp2 = nn.Dropout(p=0.2)
        self.ln2 = nn.Linear(500, 500)
        self.ReLU = nn.ReLU()

        self.dp3 = nn.Dropout(p=0.2)
        self.ln3 = nn.Linear(500, 500)
        self.ReLU = nn.ReLU()

        self.dp4 = nn.Dropout(p=0.3)
        self.ln4 = nn.Linear(500, 1)        
        #self.out = nn.Softmax(dim=1)       
        

    def forward(self, x):
        x = self.dp1(x)
        x = self.ln1(x)  
        x = self.ReLU(x)

        x = self.dp2(x)
        x = self.ln2(x)  
        x = self.ReLU(x)
        
        x = self.dp3(x)
        x = self.ln3(x)  
        x = self.ReLU(x)

        x = self.dp4(x)
        x = self.ln4(x)          
        #pred = self.out(x)
        return x



class LSTM(nn.Module):
    # https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
    
    def __init__(self, indim, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.rnn = nn.LSTM(input_size=indim, hidden_size=hidden_size, num_layers=num_layers)  # 128 256

        self.ln = nn.Linear(hidden_size, 2)        
        self.out = nn.Softmax(dim=1)       
        

    def forward(self, x):
        output, (hn, cn)  = self.rnn(x)
        x = self.ln(output)  
       
        pred = self.out(x)
        return pred

'''
An implementation of LeNet CNN architecture.
Video explanation: https://youtu.be/fcOW-Zyb5Bo
Got any questions leave a comment on youtube :)
Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-05 Initial coding
'''

import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=(2),stride=(2))
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=(5),stride=(1),padding=(0))
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=(5),stride=(1),padding=(0))
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=120, kernel_size=(5),stride=(1),padding=(0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x)) # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


