import argparse
import logging
import numpy as np
import xarray as xr
import SBCK, SBCK.tools
from pathlib import Path

from deepdown.utils.debiaser_utils import prepare_for_sbck
from deepdown.utils.data_loader import DataLoader
from deepdown.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def correct_bias(conf):
    logger.info("Loading input and targets data")

    # Load target data for the historical period
    target_data_hist = DataLoader(path_tmp=conf.path_tmp)
    target_data_hist.load(conf.period_hist_start, conf.period_hist_end,
                          conf.path_targets)

    # Load input data (e.g. climate model) for the historical period
    input_data_hist = DataLoader(path_tmp=conf.path_tmp)
    input_data_hist.load(conf.period_hist_start, conf.period_hist_end, conf.path_inputs)

    # Load input data (e.g. climate model) for the future period
    input_data_clim = DataLoader(path_tmp=conf.path_tmp)
    input_data_clim.load(conf.period_clim_start, conf.period_clim_end, conf.path_inputs)

    # Coarsen the target data to the resolution of the input data
    target_data_hist.coarsen(
        x_axis=input_data_hist.data.x, y_axis=input_data_hist.data.y,
        from_proj='CH1903_LV95', to_proj='WGS84')

    # Preparing the data for SBCK (extract and flatten the data)
    target_array_hist = []
    input_array_hist = []
    input_array_clim = []
    for var_target, var_input in zip(conf.target_vars, conf.input_vars):
        logger.info(f"Preparing data for {var_target}")

        # Prepare data
        target_array_hist_v = prepare_for_sbck(target_data_hist, var_target)
        input_array_hist_v = prepare_for_sbck(input_data_hist, var_input)
        input_array_clim_v = prepare_for_sbck(input_data_clim, var_input)

        # Reshape data (3D to 1D)
        target_array_hist_v = target_array_hist_v.flatten()
        input_array_hist_v = input_array_hist_v.flatten()
        input_array_clim_v = input_array_clim_v.flatten()

        # Append to the list
        target_array_hist.append(target_array_hist_v)
        input_array_hist.append(input_array_hist_v)
        input_array_clim.append(input_array_clim_v)

    # Stack the data
    target_array_hist = np.stack(target_array_hist, axis=1)
    input_array_hist = np.stack(input_array_hist, axis=1)
    input_array_clim = np.stack(input_array_clim, axis=1)

    # Bias correct all variables simultaneously
    logger.info(f"Processing the bias correction.")
    qm_empiric = SBCK.QM(distY0=SBCK.tools.rv_histogram,
                         distX0=SBCK.tools.rv_histogram)
    qm_empiric.fit(target_array_hist, input_array_hist)
    debiased_ts = qm_empiric.predict(input_array_clim)

    # Extract and reshape the debiased time series
    ouput_array_clim = []
    for var_out in conf.target_vars:
        debiased_ts_v = debiased_ts[:, conf.target_vars.index(var_out)]
        debiased_ts_v = debiased_ts_v.reshape(input_data_clim.data[var_out].shape)
        ouput_array_clim.append(debiased_ts_v)

    time_coords = input_data_clim.data['time']
    y_coords = input_data_clim.data['y']
    x_coords = input_data_clim.data['x']
    variables = conf.target_vars

    # Create a dictionary for the data variables
    data_vars = {var: (('time', 'y', 'x'), ouput_array_clim[i]) for i, var in
                 enumerate(variables)}

    # Create the xarray dataset
    output_data_clim = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': time_coords,
            'y': y_coords,
            'x': x_coords
        }
    )

    # Save the debiased dataset to a NetCDF file
    output_path = Path(conf.path_output)
    file_out_orig = output_path / "input_data_clim.nc"
    input_data_clim.data.to_netcdf(file_out_orig)
    file_out = output_path / "input_data_clim_debiased.nc"
    output_data_clim.to_netcdf(file_out)
    logger.info(f"Debiased dataset saved to {file_out}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--config_file", default='../config.yaml',
                           help="Path to the .yml config file")
    args = argParser.parse_args()

    logger.info("Loading configuration...")
    conf = Config(args)
    conf.print()

    logger.info("Starting bias correction")
    correct_bias(conf.get())
