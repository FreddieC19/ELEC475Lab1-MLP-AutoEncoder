import torch
from torch import nn
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer

class linearInterpolator:
    def __init__(self, model, n_steps):
        self.model = model
        self.n_steps = n_steps