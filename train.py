import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import your autoencoderMLP4Layer model here
from model import autoencoderMLP4Layer

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, args):
    print('Training...')
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device=device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()  # Removed the argument 'loss_train'

        losses_train.append(loss_train / len(train_loader))

        print('{} Epoch {}, Training loss {:.4f}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)
        ))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses_train) + 1), losses_train, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the loss plot
    plt.savefig(args.save_plot)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='MLP Autoencoder Training')
    parser.add_argument('-z', '--bottleneck', type=int, default=64, help='Bottleneck size')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('-s', '--save-model', type=str, default='MLP.pth', help='Path to save the model')
    parser.add_argument('-p', '--save-plot', type=str, default='loss.png', help='Path to save the loss plot')
    args = parser.parse_args()

    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using GPU")

    # Define data transformations and load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data/mnist', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss function, optimizer, and learning rate scheduler
    model = autoencoderMLP4Layer(N_bottleneck=args.bottleneck)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    torchsummary.summary(model, (1, 28 * 28))

    # Train the model
    train(args.epochs, optimizer, model, loss_fn, train_loader, scheduler, device, args)

    # Save the trained model
    torch.save(model.state_dict(), args.save_model)

if __name__ == '__main__':
    main()
