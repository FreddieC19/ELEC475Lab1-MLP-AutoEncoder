import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer


class addNoise:
    def __init__(self, modelPath, index):
        self.modelPath = modelPath
        self.index = index

    def applyNoise(self, modelPath, index):

        # load MNIST dataset and apply transform
        data_transform = transforms.Compose([transforms.ToTensor()])
        data_set = MNIST('./data/mnist', train=True, download=True, transform=data_transform)

        # load trained autoencoder model
        model = autoencoderMLP4Layer(N_bottleneck=8)
        model.load_state_dict(torch.load(modelPath))
        model.eval()

        # get input image from dataset and flatten it
        input_image, _ = data_set[index]
        input_image = input_image.view(-1)

        # add noise to input image
        noise_amt = 0.2
        input_noise = input_image + noise_amt * torch.randn(input_image.shape)

        # forward pass the input image through the model to get reconstructed output
        with torch.no_grad():
            reconstructed_image = model(input_noise.unsqueeze(0)).squeeze(0)

        # convert the input and reconstructed images to NumPy arrays
        input_image_numpy = input_image.view(28, 28).cpu().numpy()
        reconstructed_image_numpy = reconstructed_image.view(28, 28).cpu().numpy()  # Reshape to (28, 28)
        input_noise_numpy = input_noise.view(28, 28).cpu().numpy()

        # display the input and reconstructed images side by side
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Input Image")
        plt.imshow(input_image_numpy, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Noisy Input Image")
        plt.imshow(input_noise_numpy, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Reconstructed Image")
        plt.imshow(reconstructed_image_numpy, cmap='gray')

        plt.show()
