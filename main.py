import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer  # Import your autoencoder model class

# Define a transform to use for displaying images
display_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((28, 28)), transforms.ToTensor()])

# Load the MNIST dataset and apply the same transform as used during training
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

# Load your trained autoencoder model
model = autoencoderMLP4Layer(N_bottleneck=8)  # Adjust the bottleneck size as needed
model.load_state_dict(torch.load("MLP.8.pth"))
model.eval()

# Get user input for the index of the image to reconstruct
index = int(input("Please enter an integer between 0 and 59999: "))

# Get the input image from the dataset and flatten it
input_image, _ = train_set[index]
input_image = input_image.view(-1)  # Flatten the input

# Forward pass the input image through the model to get the reconstructed output
with torch.no_grad():
    reconstructed_image = model(input_image.unsqueeze(0)).squeeze(0)

# Convert the input and reconstructed images to NumPy arrays
input_image_numpy = input_image.view(28, 28).cpu().numpy()
reconstructed_image_numpy = reconstructed_image.view(28, 28).cpu().numpy()  # Reshape to (28, 28)

# Display the input and reconstructed images side by side
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_image_numpy, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image_numpy, cmap='gray')

plt.show()
