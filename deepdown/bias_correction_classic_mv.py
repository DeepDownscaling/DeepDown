import argparse
import logging
import copy
import numpy as np
import xarray as xr
from pathlib import Path

from deepdown.utils.debiaser_utils import extract_for_sbck, debias_with_sbck
from deepdown.utils.data_loader import DataLoader
from deepdown.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def correct_bias(conf, preload_data=True):
    """
    Correct bias in the input data using the target data.

    Parameters
    ----------
    conf : dict
        Configuration dictionary. Contains the bias correction method and paths to the
        input and target data. bias_correction_method should be one of the following:
        - 'QM': Quantile Mapping
        - 'RBC': Random Bias Correction
        - 'IdBC': Identity Bias Correction
        - 'CDFt': Quantile Mapping bias corrector, taking account of an evolution of the distribution
        - 'OTC': Optimal Transport bias Corrector
        - 'dOTC': Dynamical Optimal Transport bias Corrector
        - 'ECBC': Empirical Copula Bias Correction
        - 'QMrs': Quantile Mapping bias corrector with multivariate rankshuffle
        - 'R2D2': Non stationnary Quantile Mapping bias corrector with multivariate rankshuffle
        - 'QDM': QDM Bias correction method
        - 'MBCn': MBCn Bias correction method
        - 'MRec': MRec Bias correction method
        - 'TSMBC': Time Shifted Multivariate Bias Correction
        - 'dTSMBC': Time Shifted Multivariate Bias Correction where observations are unknown.
        - 'AR2D2': Multivariate bias correction with quantiles shuffle
    preload_data: boolean
        Whether to preload the data in memory or not.
    """
    logger.info("Loading input and targets data")

    # Load target data for the historical period
    target_data_hist = DataLoader(path_tmp=conf.path_tmp)
    target_data_hist.load(conf.period_hist_start, conf.period_hist_end,
                          conf.path_targets)

    # Load input data (e.g. climate model) for both historical and future periods
    input_data = DataLoader(path_tmp=conf.path_tmp)
    input_data.load(conf.period_hist_start, conf.period_clim_end, conf.path_inputs)

    # Get the extent of the domain of interest
    x_min, x_max, y_min, y_max = input_data.get_extent_from(
        target_data_hist, from_proj='CH1903+', to_proj='WGS84')

    # Select the domain of interest
    input_data.select_domain(x_min, x_max, y_min, y_max)
    input_data_hist = input_data
    input_data_clim = copy.deepcopy(input_data)
    input_data_hist.select_period(conf.period_hist_start, conf.period_hist_end)
    input_data_clim.select_period(conf.period_clim_start, conf.period_clim_end)
    del input_data

    # Coarsen the target data to the resolution of the input data
    target_data_hist.coarsen(
        x_axis=input_data_hist.data.x, y_axis=input_data_hist.data.y,
        from_proj='CH1903_LV95', to_proj='WGS84')

    if preload_data:
        logger.info(f"Preloading data in memory.")
        input_data_hist.data.load()
        input_data_clim.data.load()
        target_data_hist.data.load()

    # Create the output data arrays
    output_array_hist = []
    output_array_clim = []
    for i, var_out in enumerate(conf.target_vars):
        output_array_hist.append(np.ones(input_data_hist.data[var_out].shape) * np.nan)
        output_array_clim.append(np.ones(input_data_clim.data[var_out].shape) * np.nan)

    # Proceed to the point-wise bias correction
    x_axis = input_data_hist.data.x
    y_axis = input_data_hist.data.y
    for x_idx in range(x_axis.size):
        for y_idx in range(y_axis.size):
            x = float(x_axis[x_idx])
            y = float(y_axis[y_idx])
            logger.info(f"Processing point ({x:.2f}, {y:.2f})")

            # Prepare the data for SBCK
            target_array_hist = []
            input_array_hist = []
            input_array_clim = []
            for var_target, var_input in zip(conf.target_vars, conf.input_vars):
                # Convert and extract the values as numpy arrays
                target_array_hist_v = extract_for_sbck(target_data_hist, var_target, x, y)
                if target_array_hist_v is None:
                    logger.warning(f"Skipping point ({x:.2f}, {y:.2f}) for {var_target}")
                    break

                input_array_hist_v = extract_for_sbck(input_data_hist, var_input, x, y)
                input_array_clim_v = extract_for_sbck(input_data_clim, var_input, x, y)

                assert input_array_hist_v is not None, f"Missing data for {var_input} at ({x:.2f}, {y:.2f})"
                assert input_array_clim_v is not None, f"Missing data for {var_input} at ({x:.2f}, {y:.2f})"

                # Append to the list
                target_array_hist.append(target_array_hist_v)
                input_array_hist.append(input_array_hist_v)
                input_array_clim.append(input_array_clim_v)

            else:
                # Stack the data
                target_array_hist = np.stack(target_array_hist, axis=1)
                input_array_hist = np.stack(input_array_hist, axis=1)
                input_array_clim = np.stack(input_array_clim, axis=1)

                # Bias correct all variables simultaneously
                logger.info(f"Processing the bias correction with {conf.bias_correction_method}.")
                debiased_clim_ts, debiased_hist_ts = debias_with_sbck(
                    conf.bias_correction_method, input_array_clim,
                    input_array_hist, target_array_hist)

                # Store the debiased time series
                for i, var_out in enumerate(conf.target_vars):
                    output_array_hist[i][y_idx, x_idx] = debiased_hist_ts[:, conf.target_vars.index(var_out)]
                    output_array_clim[i][y_idx, x_idx] = debiased_clim_ts[:, conf.target_vars.index(var_out)]

    # Create a dictionary for the data variables
    data_vars_hist = {var: (('time', 'y', 'x'), output_array_hist[i]) for i, var in
                      enumerate(conf.target_vars)}
    data_vars_clim = {var: (('time', 'y', 'x'), output_array_clim[i]) for i, var in
                 enumerate(conf.target_vars)}

    # Create the xarray dataset
    output_data_hist = xr.Dataset(
        data_vars=data_vars_hist,
        coords={
            'time': input_data_hist.data['time'],
            'y': input_data_hist.data['y'],
            'x': input_data_hist.data['x']
        }
    )
    output_data_clim = xr.Dataset(
        data_vars=data_vars_clim,
        coords={
            'time': input_data_clim.data['time'],
            'y': input_data_clim.data['y'],
            'x': input_data_clim.data['x']
        }
    )

    # Save the debiased dataset to a NetCDF file
    output_path = Path(conf.path_output)
    output_path.mkdir(parents=True, exist_ok=True)
    target_data_hist.data.to_netcdf(output_path / "target_data_hist_original.nc")
    input_data_hist.data.to_netcdf(output_path / "input_data_hist_original.nc")
    output_data_hist.to_netcdf(output_path / "input_data_hist_debiased.nc")
    input_data_clim.data.to_netcdf(output_path / "input_data_clim_original.nc")
    output_data_clim.to_netcdf(output_path / "input_data_clim_debiased.nc")
    logger.info(f"Debiased dataset saved to {output_path}")


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
