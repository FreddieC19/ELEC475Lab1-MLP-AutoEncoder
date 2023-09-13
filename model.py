import torch
from torch import nn
from torch.utils._contextlib import F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class autoencoderMLP4Layer(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer,self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input,N2)            # input = 1x784, output = 1x392
        self.fc2 = nn.Linear(N2, N_bottleneck)      #output = 1xN
        self.fc3 = nn.Linear(N_bottleneck, N2)      # output = 1x392
        self.fc4 = nn.Linear(N2, N_output)          # output = 1x784
        self.type = 'MLP4'
        self.input_shape = (1,28*28)

    def forward(self, X):
        #encoder
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)

        #decoder
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = torch.sigmoid(X)

        return X