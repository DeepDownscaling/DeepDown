import torch
import numpy as np
import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_cuda_availability():
    """Prints whether cuda is available and the device being used."""
    print("Cuda available :", torch.cuda.is_available())
    print("Device used:", DEVICE)


def split_function(data, years_train, years_valid, years_test):

    train = data.sel(time=slice( datetime(years_train[0],1,1), datetime(years_train[1], 12, 31)))
    valid = data.sel(time=slice( datetime(years_valid[0],1,1), datetime(years_valid[1], 12, 31)))
    test = data.sel(time=slice( datetime(years_test[0],1,1), datetime(years_test[1], 12, 31)))

    return train, valid, test