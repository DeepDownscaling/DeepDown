import argparse
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from deepdown.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_bias_correction(conf):
    data_path = Path(conf.path_output)

    # Load input_data_hist_original.nc
    target_data = xr.open_dataset(data_path / "target_data_hist_original.nc")
    target_data = target_data.load()
    input_data_hist = xr.open_dataset(data_path / "input_data_hist_original.nc")
    input_data_hist = input_data_hist.load()
    output_data_hist = xr.open_dataset(data_path / "input_data_hist_debiased.nc")
    output_data_hist = output_data_hist.load()
    input_data_proj = xr.open_dataset(data_path / "input_data_proj_original.nc")
    input_data_proj = input_data_proj.load()
    output_data_proj = xr.open_dataset(data_path / "input_data_proj_debiased.nc")
    output_data_proj = output_data_proj.load()

    # Plot the maps
    for var in conf.target_vars:
        target_data_mean = target_data[var].mean(dim='time')
        input_data_hist_mean = input_data_hist[var].mean(dim='time')
        output_data_hist_mean = output_data_hist[var].mean(dim='time')
        input_data_proj_mean = input_data_proj[var].mean(dim='time')
        output_data_proj_mean = output_data_proj[var].mean(dim='time')

        min_val = min(input_data_hist_mean.min().values,
                      output_data_hist_mean.min().values,
                      input_data_proj_mean.min().values,
                      output_data_proj_mean.min().values)
        max_val = max(input_data_hist_mean.max().values,
                      output_data_hist_mean.max().values,
                      input_data_proj_mean.max().values,
                      output_data_proj_mean.max().values)

        if var == 'tp':
            cmap = 'Blues'
        elif var == 't':
            cmap = 'turbo'
        else:
            cmap = 'viridis'

        fig, ax = plt.subplots(1, 1, figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        target_data_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                              cbar_kwargs={'shrink': 0.5}, vmin=min_val,
                              vmax=max_val)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(f"Target {var}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{var}_target.png")

        fig, ax = plt.subplots(1, 1, figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        input_data_hist_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                  cbar_kwargs={'shrink': 0.5}, vmin=min_val,
                                  vmax=max_val)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(f"Input {var} (historical)")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{var}_input_hist_original.png")

        fig, ax = plt.subplots(1, 1, figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        output_data_hist_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                   cbar_kwargs={'shrink': 0.5}, vmin=min_val,
                                   vmax=max_val)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(f"Debiased {var} (historical)")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{var}_input_hist_debiased.png")

        fig, ax = plt.subplots(1, 1, figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        input_data_proj_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                  cbar_kwargs={'shrink': 0.5}, vmin=min_val,
                                  vmax=max_val)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(f"Input {var} (future)")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{var}_input_proj_original.png")

        fig, ax = plt.subplots(1, 1, figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        output_data_proj_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                   cbar_kwargs={'shrink': 0.5}, vmin=min_val,
                                   vmax=max_val)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(f"Debiased {var} (future)")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{var}_input_proj_debiased.png")

    # For each variable, do a scatter plot of the original (input_data_proj) vs. debiased data (output_data_proj)
    for var in conf.target_vars:
        fig, ax = plt.subplots()
        ax.scatter(input_data_proj[var].values.flatten(), output_data_proj[var].values.flatten(), s=1)
        ax.set_xlabel(f"Original {var}")
        ax.set_ylabel(f"Debiased {var}")
        ax.set_title(f"Original vs. Debiased {var}")
        plt.tight_layout()
        plt.show()

    target_data_hist = xr.open_dataset(data_path / "target_data_hist_original.nc")
    target_data_hist = target_data_hist.load()

    # For each variable, do a scatter plot of the original vs. target data
    for var in conf.target_vars:
        fig, ax = plt.subplots()
        ax.scatter(input_data_hist[var].values.flatten(), target_data_hist[var].values.flatten(), s=1)
        ax.set_xlabel(f"Original {var}")
        ax.set_ylabel(f"Target {var}")
        ax.set_title(f"Original vs. Target {var}")
        plt.tight_layout()
        plt.show()

    # For each variable, do a scatter plot of the debiased vs. target data
    for var in conf.target_vars:
        fig, ax = plt.subplots()
        ax.scatter(output_data_hist[var].values.flatten(), target_data_hist[var].values.flatten(), s=1)
        ax.set_xlabel(f"Debiased {var}")
        ax.set_ylabel(f"Target {var}")
        ax.set_title(f"Debiased vs. Target {var}")
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

    plot_bias_correction(conf.get())
