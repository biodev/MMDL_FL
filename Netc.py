import torch
import torch.nn as nn
from compact_bilinear_pooling import CompactBilinearPooling


class Cnn_With_Clinical_Net(nn.Module):
    def __init__(self, model, n_clin_features):
        super(Cnn_With_Clinical_Net, self).__init__()
        
        # CNN
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None  
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1] 
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 128)  

        # clinical feature
        self.clinical = nn.Linear(n_clin_features, n_clin_features) 

        self.mcb = CompactBilinearPooling(128, n_clin_features, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Linear(128, n_clin_features)  

    def forward(self, x, clinical_features):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None: 
            x = self.dense(x)
        x = self.linear(x)
        clinical = self.clinical(clinical_features)
        x = self.mcb(x, clinical)
        x = self.bn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class Net(nn.Module):  
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1] 
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None:
            x = self.dense(x)
        x = self.linear(x)
        return x
