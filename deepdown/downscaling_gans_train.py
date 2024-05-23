# Common imports
import argparse
import logging
import numpy as np

# Import torch
import torch

# Utils
from deepdown.utils.data_loader import load_target_data, load_input_data
from deepdown.utils.loss_fcts import generator_loss, discriminator_loss
from deepdown.utils.data_generators import DataGenerator
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

    logger.info("Loading input and targets data")
    input_paths = [
        conf.path_input + '/precipitation',
        conf.path_input + '/temperature',
        conf.path_input + '/max_temperature/',
        conf.path_input + '/min_temperature/'
    ]
    
    target_paths = [
        conf.path_target + '/RhiresD_v2.0_swiss.lv95/',
        conf.path_target + '/TabsD_v2.0_swiss.lv95/',
        conf.path_target + '/TmaxD_v2.0_swiss.lv95/',
        conf.path_target + '/TminD_v2.0_swiss.lv95/'
    ]
  
    # Load target data
    target = load_target_data(conf.date_start, conf.date_end, target_paths,
                              path_tmp=conf.path_tmp)

    input_data = load_input_data(
        date_start=conf.date_start, date_end=conf.date_end, levels=conf.levels,
        resol_low=conf.resol_low, x_axis=target.x, y_axis=target.y,
        paths=input_paths, path_dem=conf.path_dem, dump_data_to_pickle=True,
        path_tmp=conf.path_tmp)

    # Split the data
    x_train = split_data(input_data, conf.years_train)
    x_valid = split_data(input_data, conf.years_valid)
    x_test = split_data(input_data, conf.years_test)
    y_train = split_data(target, conf.years_train)
    y_valid = split_data(target, conf.years_valid)
    y_test = split_data(target, conf.years_test)

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
