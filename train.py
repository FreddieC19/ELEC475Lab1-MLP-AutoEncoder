import datetime
import torch.nn as nn
from torchsummary import summary
import argparse
import torch.optim as optimizer
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torch.optim.lr_scheduler as scheduler



def main():
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='Training script for the model')
    parser.add_argument('-z', type=int, help='Value for -z')
    parser.add_argument('-e', type=int, help='Value for -e')
    parser.add_argument('-b', type=int, help='Value for -b')
    parser.add_argument('-s', type=str, help='Value for -s')
    parser.add_argument('-p', type=str, help='Value for -p')
    args = parser.parse_args()

    # Extract the parsed arguments
    z_value = args.z
    e_value = args.e
    b_value = args.b
    s_value = args.s
    p_value = args.p

    print('Parsed arguments:')
    print('-z:', z_value)
    print('-e:', e_value)
    print('-b:', b_value)
    print('-s:', s_value)
    print('-p:', p_value)

    retString = [z_value, e_value, b_value, s_value, p_value]
    return retString


if __name__ == '__main__':
    main()

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('Training...')
    model.train() #keep track of gradient for backtracking
    losses_train = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs in train_loader:           # imgs is a minibatch of data
            imgs = imgs.to(device=device)   # use cpu or gpu
            outputs = model(imgs)           # forward propogation through network
            loss = loss_fn(outputs, imgs)   # calculate loss
            optimizer.zero_grad()           # reset optimizer gradients to zero
            loss.backward()                 # calculate loss gradients
            optimizer.step()                # iterate the optimization, based on loss gradients
            loss_train += loss.item         # update value of losses


        scheduler.step(loss_train)  #update some optimization hyperparameters

        losses_train += [loss_train/len(train_loader)]  #update value of losses

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)
        ))

        #summary(model, input_size=len(train_loader))

        #to make module callable, if __name__=main:

values = main()
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)


train(values[1], optimizer, 'model.py', nn.MSELoss, train_set, scheduler, 'cpu')
