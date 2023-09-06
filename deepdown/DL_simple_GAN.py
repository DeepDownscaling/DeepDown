# Common imports
import argparse
import sys
# Add the parent directory to sys.path
sys.path.append('/storage/homefs/no21h426/DL-downscaling/')
import os
import yaml
import math
import warnings
import numpy as np
import xarray as xr
import dask
from time import time

# Import torch
import torch
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, sampler, TensorDataset
from torch.utils.data import sampler

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T


# Utils
from utils.data_loader import *
from utils.utils_plot import *
from utils.utils_loss import *
from utils.helpers import *
from utils.Datagenerators import *
from utils.SRGAN import *
from deepdown.train_srgan import *

# Try dask.distributed and see if the performance improves...
from dask.distributed import Client
#c = Client(n_workers=os.cpu_count()-2, threads_per_worker=1)

#warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file") 

print("Cuda Avaliable :", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(config):
    # Paths
    PATH_DEM = config['PATH_DEM']
    PATH_ERA5_025 = config['PATH_ERA5_025']  # Original ERA5 0.25°
    PATH_ERA5_100 = config['PATH_ERA5_100']  # ERA5 1°
    PATH_MCH = config['PATH_MeteoSwiss']  
    
    NUM_CHANNELS_IN = config['NUM_CHANNELS_IN']
    NUM_CHANNELS_OUT = config['NUM_CHANNELS_OUT']
    lr = config['lr']

    # Data options
    DATE_START = config['DATE_START']
    DATE_END = config['DATE_END']
    YY_TRAIN = config['YY_TRAIN']  
    YY_TEST = config['YY_TEST']  
    LEVELS = config['LEVELS'] 
    RESOL_LOW = config['RESOL_LOW'] 
    INPUT_VARIABLES = config['INPUT_VARIABLES']
    INPUT_PATHS = [PATH_ERA5_025 + '/precipitation', PATH_ERA5_025 + '/temperature']
    DUMP_DATA_TO_PICKLE = config['DUMP_DATA_TO_PICKLE']

    # Crop on a smaller region
    DO_CROP = config['DO_CROP']
    # I reduce the area of crop now, to avoid NA
    CROP_X = [2700000, 2760000]  # with NAN: [2720000, 2770000]
    CROP_Y = [1190000, 1260000]  # with NAN: [1290000, 1320000]

    device = config['device']
    dtype = config['dtype'] 
    # Hyperparameters
    BATCH_SIZE = 32
    h, w = config['lowres_shape']

    # Load target data
    target = load_target_data(DATE_START, DATE_END, PATH_MCH)

    # Extract the axes of the final target domain based on temperature 
    x_axis = target.TabsD.x
    y_axis = target.TabsD.y


    input_data = load_input_data(DATE_START, DATE_END, PATH_DEM, INPUT_VARIABLES, INPUT_PATHS, 
                             LEVELS, RESOL_LOW, x_axis, y_axis)


    if DO_CROP:
        input_data = input_data.sel(x=slice(min(CROP_X), max(CROP_X)), y=slice(max(CROP_Y), min(CROP_Y)))
        target = target.sel(x=slice(min(CROP_X), max(CROP_X)), y=slice(max(CROP_Y), min(CROP_Y)))

    # Split the data
    x_train = input_data.sel(time=slice('1999', '2011')) 
    x_valid = input_data.sel(time=slice('2012', '2015')) 
    x_test = input_data.sel(time=slice('2016', '2021'))

    y_train = target.sel(time=slice('1999', '2011'))
    y_valid = target.sel(time=slice('2012', '2005'))
    y_test = target.sel(time=slice('2006', '2011'))


    # Select the variables to use as input and output
    input_vars = {'topo' : None, 'tp': None, 't': LEVELS}
    output_vars = ['RhiresD', 'TabsD'] #['RhiresD', 'TabsD', 'TmaxD', 'TminD']


    training_set = DataGenerator(x_train, y_train, input_vars, output_vars)
    loader_train = torch.utils.data.DataLoader(training_set, batch_size=32)

    # Validation
    valid_set = DataGenerator(x_valid, y_valid, input_vars, output_vars, shuffle=False, mean=training_set.mean, std=training_set.std)
    loader_val = torch.utils.data.DataLoader(valid_set, batch_size=32)

    # Test
    test_set = DataGenerator(x_test, y_test, input_vars, output_vars, shuffle=False, mean=training_set.mean, std=training_set.std)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=32)


    # Check to make sure the range on the input and output images is correct, and they're the correct shape
    testx, testy = training_set.__getitem__(3)
    print("x shape: ", testx.shape)
    print("y shape: ", testy.shape)
    torch.cuda.empty_cache()


    D = Discriminator(num_channels=NUM_CHANNELS_OUT, H=h,W=w) 
    G = Generator(NUM_CHANNELS_IN, input_size = config['lowres_shape'], output_channels=NUM_CHANNELS_OUT)

    # Define optimizer for discriminator
    D_solver = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # Define optimizer for generator
    G_solver = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


    num_epochs=config['num_epochs']
    G_iters=config['G_iters']
    dtype = torch.float32
    
    print('train the SRGAN')
    train_srgan(loader_train, D, G, D_solver, G_solver, discriminator_loss, generator_loss, device, dtype, G_iters=1, show_every=250, num_epochs=5)
    
    
    
    
if __name__ == "__main__":

    args = argParser.parse_args()
  
    config = read_config(args.config_file)

    print_config(config)
    main(config)





