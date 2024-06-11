import argparse
import logging
import numpy as np
import xarray as xr
import os

from ibicus.debias import QuantileMapping
from deepdown.utils.debiaser import _step_to_impute_values
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
    
    # Create a new dataset for debiased variables
    # debiased_data = xr.Dataset(coords=input_data_clim.data.coords)
    # Bias correct each input variable
    for var_target, var_input in zip(conf.target_vars, conf.input_vars):
        logger.info(f"Bias correcting variable {var_target}")
        # Map to the ibicus variable names
        var_ibicus = var_target
        if var_target == 'tp':
            var_ibicus = 'pr'
        elif var_target == 't':
            var_ibicus = 'tas'
        elif var_target == 't_min':
            var_ibicus = 'tasmin'
        elif var_target == 't_max':
            var_ibicus = 'tasmax'

        target_array_hist = target_data_hist.data[var_target].values
        input_array_hist = input_data_hist.data[var_input].values
        input_array_clim = input_data_clim.data[var_input].values

        # Change units
        if var_target == 'tp':
            # mm/day to kg/m^2/s
            target_array_hist /= 86400
            input_array_hist /= 86400
            input_array_clim /= 86400

        elif var_target in ['t', 't_min', 't_max']:
            # Degree Celsius to Kelvin
            target_array_hist += 273.15 # I think this's only applies to the target
            # input_array_hist += 273.15
            # input_array_clim += 273.15
        if conf.imputed_method is not None:
            logger.info("imputed values")
            if var_ibicus == 'tp':
                iecdf_method = 'averaged_inverted_cdf' #'closest_observation'
            else:
                iecdf_method = 'linear'
            target_array_hist = _step_to_impute_values(target_array_hist, iecdf_method=iecdf_method)
            input_array_hist = _step_to_impute_values(input_array_hist, iecdf_method=iecdf_method)
            input_array_clim = _step_to_impute_values(input_array_clim, iecdf_method=iecdf_method)
        else:
            logger.info("replace with NaN")
            # Replace NaNs with zeros
            target_array_hist = np.nan_to_num(target_array_hist)
            input_array_hist = np.nan_to_num(input_array_hist)
            input_array_clim = np.nan_to_num(input_array_clim)


        debiaser = QuantileMapping.from_variable(var_ibicus)
        debiased_ts = debiaser.apply(
            target_array_hist,
            input_array_hist,
            input_array_clim)
    

        debiased_var = f"{var_input}_deb"
        input_data_clim.data = input_data_clim.data.assign(**{debiased_var: (('time', 'y', 'x'), debiased_ts)})


    # Save the debiased dataset to a NetCDF file

    output_path = conf.path_output
    file_out = os.path.join(output_path, "input_data_clim_zeroimput_debiased.nc")
    input_data_clim.data.to_netcdf(file_out)
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
