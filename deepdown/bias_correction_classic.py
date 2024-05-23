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

    # Load target data
    target_data = load_target_data(
        conf.date_start, conf.date_end, conf.path_targets,
        path_tmp=conf.path_tmp)

    # Coarsen the target data to the resolution of the input data
    target_data.coarsen(...)

    # Load input data
    input_data = load_input_data(
        date_start=conf.date_start, date_end=conf.date_end, levels=conf.levels,
        resol_low=conf.resol_low, x_axis=target_data.x, y_axis=target_data.y,
        paths=conf.path_inputs, path_dem=conf.path_dem, dump_data_to_pickle=True,
        path_tmp=conf.path_tmp)

    # Bias correct each input variable
    for var_target, var_input in zip(conf.target_vars, conf.input_vars):
        logger.info(f"Bias correcting variable {var_target}")
        debiaser = QuantileMapping.from_variable(var_target)
        debiased_ts = debiaser.apply(
            target_data[var_target],
            input_data[var_input],
            input_data[var_input])
        input_data[var_input] = debiased_ts


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
