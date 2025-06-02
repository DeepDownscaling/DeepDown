import os
import warnings
import numpy as np
import requests
from PIL import Image

import torch
import torch.utils.data as data

from transformers import Swin2SRConfig, Swin2SRModel
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# Utils
from deepdown.utils.data_loader import *
from deepdown.utils.data_generator import *
from deepdown.utils.utils import *
from deepdown.constants import *

# Try dask.distributed and see if the performance improves...
from dask.distributed import Client
c = Client(n_workers=os.cpu_count()-2, threads_per_worker=1)

warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

print_cuda_availability()

target = load_target_data(DATE_START, DATE_END, path_targets)
x_axis = target.TabsD.x
y_axis = target.TabsD.y

input_data = load_input_data(DATE_START, DATE_END, PATH_DEM, INPUT_VARIABLES,
                             INPUT_PATHS,
                             LEVELS, RESOL_LOW, x_axis, y_axis)

if DO_CROP:
    input_data = input_data.sel(x=slice(min(CROP_X), max(CROP_X)),
                                y=slice(max(CROP_Y), min(CROP_Y)))
    target = target.sel(x=slice(min(CROP_X), max(CROP_X)),
                        y=slice(max(CROP_Y), min(CROP_Y)))

# Split the data (small data for testing purposes)
x_train = input_data.sel(time=slice('1999', '2011'))
x_valid = input_data.sel(time=slice('2012', '2015'))
x_test = input_data.sel(time=slice('2016', '2021'))

y_train = target.sel(time=slice('1999', '2011'))
y_valid = target.sel(time=slice('2012', '2015'))
y_test = target.sel(time=slice('2006', '2011'))

# Select the variables to use as input and output
input_vars = {'topo': None, 'tp': None, 't': LEVELS}
output_vars = ['RhiresD', 'TabsD']  # ['RhiresD', 'TabsD', 'TmaxD', 'TminD']

training_set = DataGenerator(x_train, y_train, input_vars, output_vars)
loader_train = torch.utils.data.DataLoader(training_set, batch_size=32)

# Validation
valid_set = DataGenerator(x_valid, y_valid, input_vars, output_vars, shuffle=False,
                          x_mean=training_set.x_mean, x_std=training_set.x_std)
loader_val = torch.utils.data.DataLoader(valid_set, batch_size=32)

# Test
test_set = DataGenerator(x_test, y_test, input_vars, output_vars, shuffle=False,
                         x_mean=training_set.x_mean, x_std=training_set.x_std)
loader_test = torch.utils.data.DataLoader(test_set, batch_size=32)

torch.cuda.empty_cache()

# Initializing a Swin2SR caidas/swin2sr-classicalsr-x2-64 style configuration
configuration = Swin2SRConfig()

# Initializing a model (with random weights) from the caidas/swin2sr-classicalsr-x2-64 style configuration
#model = Swin2SRModel(configuration)

# Accessing the model configuration
#configuration = model.config

configuration

processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

url = "https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# prepare image for the model
inputs = processor(image, return_tensors="pt")

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.moveaxis(output, source=0, destination=-1)
output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
# you can visualize `output` with `Image.fromarray`
