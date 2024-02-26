import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_cuda_availability():
    """Prints whether cuda is available and the device being used."""
    print("Cuda available :", torch.cuda.is_available())
    print(DEVICE)

