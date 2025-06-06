import argparse
import logging
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

from deepdown.utils.debiaser_utils import prepare_for_sbck, debias_with_sbck
from deepdown.utils.data_loader import DataLoader
from deepdown.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_maps(conf):
    # Load target data for the historical period
    target_data_hist = DataLoader(path_tmp=conf.path_tmp)
    target_data_hist.load(conf.period_hist_start, conf.period_hist_end,
                          conf.path_targets)

    # Load input data (e.g. climate model) for the historical period
    input_data_hist = DataLoader(path_tmp=conf.path_tmp)
    input_data_hist.load(conf.period_hist_start, conf.period_hist_end, conf.path_inputs)

    # Load input data (e.g. climate model) for the future period
    input_data_proj = DataLoader(path_tmp=conf.path_tmp)
    input_data_proj.load(conf.period_proj_start, conf.period_proj_end, conf.path_inputs)

    # Coarsen the target data to the resolution of the input data
    target_data_hist_coarsened = DataLoader(path_tmp=conf.path_tmp)
    target_data_hist_coarsened.load(conf.period_hist_start, conf.period_hist_end,
                                    conf.path_targets)
    target_data_hist_coarsened.coarsen(
        x_axis=input_data_hist.data.x, y_axis=input_data_hist.data.y,
        from_proj='CH1903_LV95', to_proj='WGS84')

    # Plot the maps
    for var in conf.target_vars:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        mean_val = target_data_hist.data[var].mean(dim='time')
        ax.imshow(mean_val, cmap='viridis')
        # ax[0].set_title(f"Input {var}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        mean_val = input_data_hist.data[var].mean(dim='time')
        ax.imshow(mean_val, cmap='viridis')
        # ax[0].set_title(f"Input {var}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        mean_val = target_data_hist_coarsened.data[var].mean(dim='time')
        ax.imshow(mean_val, cmap='viridis')
        # ax[0].set_title(f"Input {var}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--config_file", default='../config.yaml',
                           help="Path to the .yml config file")
    args = argParser.parse_args()

    logger.info("Loading configuration...")
    conf = Config(args)
    conf.print()

    logger.info("Starting bias correction")
    plot_maps(conf.get())
