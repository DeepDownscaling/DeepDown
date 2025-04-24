import os
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from deepdown.config import Config


def plot_maps_for_bc_method(conf, files, method, var):
    models = conf.RCMs
    path_output = conf.path_output

    # Create figure with subplots for all models and data types
    fig, axes = plt.subplots(len(models), len(files),
                             figsize=(len(files) * 4, len(models) * 4),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for row, model in enumerate(models):
        data_path = os.path.join(path_output, method, model)
        for col, data_type in enumerate(files):
            file_path = os.path.join(data_path, data_type)

            if os.path.exists(file_path):
                print(f"Loading: {file_path}")
                ds = xr.open_dataset(file_path).load()

                if var in ds:
                    data = ds[var].mean(dim="time")  # Compute time mean

                    # Plot on corresponding subplot
                    ax = axes[row, col]
                    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                              add_colorbar=(col == len(files) - 1))

                    ax.set_title(f"{model}\n{data_type.split('.')[0]}", fontsize=10)
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(path_output, f"maps_for_bc_method_{method}_{var}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_maps_for_rcm(conf, files, model, var):
    methods = conf.bc_methods
    path_output = conf.path_output

    # Create figure with subplots for all methods and data types
    fig, axes = plt.subplots(len(methods), len(files),
                             figsize=(len(files) * 4, len(methods) * 4),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for row, method in enumerate(methods):
        data_path = os.path.join(path_output, method, model)
        for col, data_type in enumerate(files):
            file_path = os.path.join(data_path, data_type)

            if os.path.exists(file_path):
                print(f"Loading: {file_path}")
                ds = xr.open_dataset(file_path).load()

                if var in ds:
                    data = ds[var].mean(dim="time")  # Compute time mean

                    # Plot on corresponding subplot
                    ax = axes[row, col]
                    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                              add_colorbar=(col == len(files) - 1))

                    ax.set_title(f"{method}\n{data_type.split('.')[0]}", fontsize=10)
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(path_output, f"maps_for_rcm_{model}_{var}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    # Load the configuration
    conf = Config().get()
    models = conf.RCMs
    methods = conf.bc_methods

    files = [
        "target_data_hist_original.nc",
        "input_data_hist_original.nc",
        "input_data_hist_debiased.nc",
        "input_data_clim_original.nc",
        "input_data_clim_debiased.nc"
    ]

    for method in methods:
        for var in conf.input_vars:
            plot_maps_for_bc_method(conf, files, method, var)

    for model in models:
        for var in conf.input_vars:
            plot_maps_for_rcm(conf, files, model, var)


if __name__ == "__main__":
    main()
