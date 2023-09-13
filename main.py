import torch
from torch import nn
from torch.utils._contextlib import F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

index = int(input("Please enter an integer between 0 and 59999: "))

plt.imshow(train_set.data[index], cmap='gray')
plt.show()

