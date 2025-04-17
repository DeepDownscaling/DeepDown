import os
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def process_files(data_path, output_dir, var):
    """Loads NetCDF files, plots all models in a single figure, and saves it."""
    models = [
        "ALADIN63_CNRM-CM5",
        "ALADIN63_MPI-ESM-LR",
        "CCLM4-8-17_MIROC5",
        "CCLM4-8-17_MPI-ESM-LR",
        "REMO2015_MIROC5"
    ]

    data_types = [
        "target_data_hist_original.nc",
        "input_data_hist_original.nc",
        "input_data_hist_debiased.nc",
        "input_data_clim_original.nc",
        "input_data_clim_debiased.nc"
    ]

    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots for all models and data types
    fig, axes = plt.subplots(len(models), len(data_types), figsize=(20, 15),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for row, model in enumerate(models):
        for col, data_type in enumerate(data_types):
            file_path = os.path.join(data_path, model, data_type)

            if os.path.exists(file_path):
                print(f"Loading: {file_path}")
                ds = xr.open_dataset(file_path).load()

                if var in ds:
                    data = ds[var].mean(dim="time")  # Compute time mean

                    # Plot on corresponding subplot
                    ax = axes[row, col]
                    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm", add_colorbar=(col == len(data_types) - 1))

                    ax.set_title(f"{model}\n{data_type.split('.')[0]}", fontsize=10)
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(output_dir, f"all_models_{var}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Process NetCDF files and save multi-model plots.")
    parser.add_argument("--data_path", type=str, default="/storage/workspaces/giub_hydro/hydro/data/tmp2/QM/")
    parser.add_argument("--output_dir", type=str, default="/storage/workspaces/giub_hydro/hydro/data/tmp2/QM/")
    parser.add_argument("--var", type=str, required=True, choices=["tp", "t"], help="Variable to plot ('tp' or 't').")

    args = parser.parse_args()
    
    process_files(args.data_path, args.output_dir, args.var)

if __name__ == "__main__":
    main()
