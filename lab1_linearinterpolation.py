import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer
from torch import nn

class linearInterpolator:
    def __init__(self, modelPath, n_steps):
        self.model = autoencoderMLP4Layer()
        self.model.load_state_dict(torch.load(modelPath))
        self.model.eval()
        self.n_steps = n_steps

    def interpolator(self, index1, index2):
        # load the MNIST dataset and apply same transform as used during training
        transform = transforms.Compose([transforms.ToTensor()])
        data_set = MNIST('./data/mnist', train=True, download=True, transform=transform)

        input_image1, _ = data_set[index1]
        input_image1 = input_image1.view(-1)

        input_image2, _ = data_set[index2]
        input_image2 = input_image2.view(-1)

        bottleneck1 = self.model.encode(input_image1)
        bottleneck2 = self.model.encode(input_image2)

        interpolatedTensors = []
        for step in range(self.n_steps):
            alpha = step / (self.n_steps -1)
            interpolatedBottleneck = alpha*bottleneck1 + (1-alpha)*bottleneck2
            interpolatedTensors.append(interpolatedBottleneck)



        decoded_images = [self.model.decode(tensor) for tensor in interpolatedTensors]

        decoded_images.insert(0,input_image2)
        decoded_images.append(input_image1)

        plt.figure(figsize=(8, 2))
        for i, decoded_image in enumerate(decoded_images):
            plt.subplot(1, self.n_steps+2, i + 1)
            plt.imshow(decoded_image.view(28, 28).detach().numpy(), cmap='gray')
            if i == 0: plt.title('Image 1')
            elif i == self.n_steps+1: plt.title('Image 2')
            else: plt.title(f'Step {i}')

        plt.show()