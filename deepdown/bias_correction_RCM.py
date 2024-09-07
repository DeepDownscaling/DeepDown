import argparse
import logging
from pathlib import Path
import shutil

from deepdown.bias_correction_classic import correct_bias
from deepdown.config import Config
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--config_file", default='../config.yaml', help="Path to the .yml config file")
    args = argParser.parse_args()

    logger.info("Loading configuration...")
    conf = Config(args)
    conf.print()

    base_config = conf.get()

    for rcm_model in base_config['RCM']:


        # Create a copy of the configuration and update path_inputs
        config_dict = deepcopy(base_config)
        config_dict['path_inputs'] = [
            f"/storage/workspaces/giub_hydro/hydro/data/RCM/regular-g0d11/precipitation/{rcm_model}/",
            f"/storage/workspaces/giub_hydro/hydro/data/RCM/regular-g0d11/temperature/{rcm_model}/"
        ]
        
        # Define a model-specific output path
        model_output_dir = Path(f"/storage/workspaces/giub_hydro/hydro/data/tmp/RCM/{rcm_model}/")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        config_dict['path_output'] = str(model_output_dir)  # Convert to string for compatibility
        
        logger.info(f"Input paths updated to: {config_dict['path_inputs']}")
        logger.info(f"Output path set to: {model_output_dir}")

        logger.info("Starting bias correction")
        correct_bias(config_dict)