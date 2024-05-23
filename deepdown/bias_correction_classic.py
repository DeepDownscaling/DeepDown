# Common imports
import argparse
import logging

from ibicus.debias import QuantileMapping

# Utils
from deepdown.utils.data_loader import load_target_data, load_input_data
from deepdown.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def correct_bias(conf):
    logger.info("Loading input and targets data")

    # Load target data for the historical period
    target_data_hist = load_target_data(
        conf.period_hist_start, conf.period_hist_end, conf.path_targets,
        path_tmp=conf.path_tmp)

    # Coarsen the target data to the resolution of the input data
    target_data_hist = coarsen_data(target_data_hist)

    # Load input data (e.g. climate model) for the historical period
    input_data_hist = load_input_data(
        date_start=conf.period_hist_start, date_end=conf.period_hist_end,
        levels=conf.levels, resol_low=conf.resol_low, x_axis=target_data_hist.x,
        y_axis=target_data_hist.y, paths=conf.path_inputs, path_dem=conf.path_dem,
        dump_data_to_pickle=True, path_tmp=conf.path_tmp)

    # Load input data (e.g. climate model) for the future period
    input_data_clim = load_input_data(
        date_start=conf.period_clim_start, date_end=conf.period_clim_end,
        levels=conf.levels, resol_low=conf.resol_low, x_axis=target_data_hist.x,
        y_axis=target_data_hist.y, paths=conf.path_inputs, path_dem=conf.path_dem,
        dump_data_to_pickle=True, path_tmp=conf.path_tmp)

    # Bias correct each input variable
    for var_target, var_input in zip(conf.target_vars, conf.input_vars):
        logger.info(f"Bias correcting variable {var_target}")
        debiaser = QuantileMapping.from_variable(var_target)
        debiased_ts = debiaser.apply(
            target_data_hist[var_target],
            input_data_hist[var_input],
            input_data_clim[var_input])
        input_data_clim[var_input] = debiased_ts


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
