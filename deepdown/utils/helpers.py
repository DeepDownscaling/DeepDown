import torch
import numpy as np
import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_cuda_availability():
    """Prints whether cuda is available and the device being used."""
    print("Cuda available :", torch.cuda.is_available())
    print("Device used:", DEVICE)


def split_data(data, years):
    subset = data.sel(time=slice(datetime(years[0], 1, 1),
                                 datetime(years[1], 12, 31)))

    return subset
