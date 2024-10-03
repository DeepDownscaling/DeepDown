import argparse
import logging
from pathlib import Path

from ibicus.debias import QuantileMapping
from deepdown.utils.debiaser_utils import get_ibicus_var_name, prepare_for_ibicus
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
    # just for testing:
    target_data_clim = DataLoader(path_tmp=conf.path_tmp)
    target_data_clim.load(conf.period_clim_start, conf.period_clim_end,
                          conf.path_targets)
    target_data_clim.coarsen(
        x_axis=input_data_clim.data.x, y_axis=input_data_clim.data.y,
        from_proj='CH1903_LV95', to_proj='WGS84')


    # Bias correct each input variable
    for var_target, var_input in zip(conf.target_vars, conf.input_vars):
        logger.info(f"Bias correcting variable {var_target}")
        # Map to the ibicus variable names
        var_ibicus = get_ibicus_var_name(var_target)

        # Prepare data for ibicus
        target_array_hist = prepare_for_ibicus(target_data_hist, var_target)
        input_array_hist = prepare_for_ibicus(input_data_hist, var_input)
        input_array_clim = prepare_for_ibicus(input_data_clim, var_input)

        debiaser = QuantileMapping.from_variable(var_ibicus)
        debiased_ts = debiaser.apply(
            target_array_hist,
            input_array_hist,
            input_array_clim)

        debiased_var = f"{var_input}_deb"
        input_data_clim.data = input_data_clim.data.assign(
            **{debiased_var: (('time', 'y', 'x'), debiased_ts)})

    # Save the debiased dataset to a NetCDF file
    output_path = Path(conf.path_output)
    file_out = output_path/"input_data_clim_debiased.nc"
    # to save RCM
 
    if '_NCProperties' in input_data_clim.data.attrs:
        logger.info(f"removing attributes")
        del input_data_clim.data.attrs['_NCProperties']

    input_data_clim.data.to_netcdf(file_out)
    logger.info(f"Debiased dataset saved to {file_out}")
    file_test = output_path/"target_clim_period.nc"
    target_data_clim.data.to_netcdf(file_test)


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
