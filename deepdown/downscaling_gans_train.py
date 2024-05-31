# Common imports
import argparse
import logging
import numpy as np

# Import torch
import torch

# Utils
from deepdown.utils.data_loader import DataLoader
from deepdown.utils.data_generator import DataGenerator
from deepdown.utils.loss_fcts import generator_loss, discriminator_loss
from deepdown.utils.helpers import print_cuda_availability, split_data
from deepdown.models.srgan import Generator, Discriminator
from deepdown.config import Config
from deepdown.models.srgan_train import srgan_train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To make this notebook's output stable across runs
np.random.seed(42)

# Check if CUDA is available
print_cuda_availability()


def train(conf):
    # Load data
    logger.info("Loading input and targets data")
    target_data = DataLoader(path_tmp=conf.path_tmp)
    target_data.load(conf.date_start, conf.date_end, conf.path_targets)

    input_data = DataLoader(path_tmp=conf.path_tmp)
    input_data.load(conf.date_start, conf.date_end, conf.path_inputs)
    input_data.regrid(x_axis=target_data.data.x, y_axis=target_data.data.y,
                      from_proj='WGS84', to_proj='CH1903_LV95', method='nearest')

    # Split the data
    x_train = split_data(input_data.data, conf.years_train)
    x_valid = split_data(input_data.data, conf.years_valid)
    x_test = split_data(input_data.data, conf.years_test)
    y_train = split_data(target_data.data, conf.years_train)
    y_valid = split_data(target_data.data, conf.years_valid)
    y_test = split_data(target_data.data, conf.years_test)

    logger.info("Creating data loaders")
    training_set = DataGenerator(
        x_train, y_train, conf.input_vars, conf.target_vars, do_crop=conf.do_crop,
        crop_x=conf.crop_x, crop_y=conf.crop_y, shuffle=True, tp_log=None)
    loader_train = torch.utils.data.DataLoader(training_set, batch_size=conf.batch_size)
    valid_set = DataGenerator(
        x_valid, y_valid, conf.input_vars, conf.target_vars, do_crop=conf.do_crop,
        crop_x=conf.crop_x, crop_y=conf.crop_y, shuffle=False,
        x_mean=training_set.x_mean, x_std=training_set.x_std)
    loader_val = torch.utils.data.DataLoader(valid_set, batch_size=conf.batch_size)
    test_set = DataGenerator(
        x_test, y_test, conf.input_vars, conf.target_vars, do_crop=conf.do_crop,
        crop_x=conf.crop_x, crop_y=conf.crop_y, shuffle=False,
        x_mean=training_set.x_mean, x_std=training_set.x_std)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=conf.batch_size)

    # Initializing models
    logger.info("Initializing models and optimizers")
    D = Discriminator(num_channels=conf.n_channels_out,
                      H=conf.input_size[0],
                      W=conf.input_size[1])
    G = Generator(num_channels=conf.n_channels_in,
                  input_size=conf.input_size,
                  output_channels=conf.n_channels_out)

    # Define optimizer for discriminator
    D_solver = torch.optim.Adam(D.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))
    # Define optimizer for generator
    G_solver = torch.optim.Adam(G.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))

    logger.info('Training the SRGAN')
    srgan_train(loader_train, D, G, D_solver, G_solver, discriminator_loss,
                generator_loss, G_iters=1, show_every=250, num_epochs=conf.num_epochs)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--config_file", default='../config.yaml',
                           help="Path to the .yml config file")
    args = argParser.parse_args()

    logger.info("Loading configuration...")
    conf = Config(args)
    conf.print()

    logger.info("Starting training process")
    train(conf.get())
