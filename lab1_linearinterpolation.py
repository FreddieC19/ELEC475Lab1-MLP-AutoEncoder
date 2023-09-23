import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer


class linearInterpolator:
    def __init__(self, modelPath, numSteps, index1, index2):
        self.modelPath = modelPath
        self.numSteps = numSteps
        self.index1 = index1
        self.index2 = index2

    def interpolator(self, modelPath, numSteps, index1, index2):
        # load MNIST dataset and apply transform
        data_transform = transforms.Compose([transforms.ToTensor()])
        data_set = MNIST('./data/mnist', train=True, download=True, transform=data_transform)

        # load trained autoencoder model
        model = autoencoderMLP4Layer(N_bottleneck=8)
        model.load_state_dict(torch.load(modelPath))
        model.eval()

        # get input image 1 from dataset and flatten it
        inputImage1, _ = data_set[index1]
        inputImage1 = inputImage1.view(-1)

        # get input image 2 from dataset and flatten it
        inputImage2, _ = data_set[index2]
        inputImage2 = inputImage2.view(-1)

        # encode both images
        bottleneck1 = model.encode(inputImage1)
        bottleneck2 = model.encode(inputImage2)

        # interpolate the bottlenecks and store them in an array
        interpolatedTensors = []
        for step in range(numSteps):
            percent = step / (numSteps - 1)
            interpolatedBottleneck = percent * bottleneck1 + (1 - percent) * bottleneck2
            interpolatedTensors.append(interpolatedBottleneck)

        # decode images and store them in new array
        decodedImages = [model.decode(tensor) for tensor in interpolatedTensors]

        # add original images to decoded images array for the plot
        decodedImages.insert(0, inputImage2)
        decodedImages.append(inputImage1)

        # display original images and interpolated images
        plt.figure(figsize=(8, 2))
        for i, decodedImage in enumerate(decodedImages):
            plt.subplot(1, numSteps + 2, i + 1)
            plt.imshow(decodedImage.view(28, 28).detach().numpy(), cmap='gray')
            if i == 0:
                plt.title('Image 1')
            elif i == numSteps + 1:
                plt.title('Image 2')
            else:
                plt.title(f'Step {i}')

        plt.show()
