import torch
from torch import nn

class autoencoderMLP4Layer(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)           #input = 1x784, output = 1x392
        self.fc2 = nn.Linear(N2, N_bottleneck)      #output = 1xN
        self.fc3 = nn.Linear(N_bottleneck, N2)      #output = 1x392
        self.fc4 = nn.Linear(N2, N_output)          #output = 1x784
        self.input_shape = (1, 28 * 28)

    def forward(self, X):
        #encoder
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))

        #decoder
        X = torch.relu(self.fc3(X))
        X = torch.sigmoid(self.fc4(X))

        return X
