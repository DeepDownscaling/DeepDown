# Compare the bias correction methods from SBCK. This script computes the bias correction for different RCM
# outputs and saves the results as netCDF files. The script bias_correction_plot_comparison_methods_sbck can be used
# to plot the results.

import sys
from pathlib import Path
from deepdown.config import Config
from deepdown.bias_correction_sbck import run_bias_correction


def assess_single(index):
    # Load the configuration
    conf = Config().get()
    models = conf.RCMs
    methods = conf.bc_methods
    path_inputs = conf.path_inputs
    path_output = conf.path_output

    # All model-method pairs
    job_list = [(model, method) for model in models for method in methods]

    # Select the model-method pair based on the index
    model, method = job_list[index]
    print(f"Running bias correction for {model} using {method}")

    # Adjust the input and output paths for the current model
    conf.path_inputs = [
        f'{path_inputs[0]}/{model}',
        f'{path_inputs[1]}/{model}'
    ]
    conf.path_output = f'{path_output}/{method}/{model}'

    # If the output directory exists, skip the setting
    if Path(conf.path_output).exists():
        print(f"Output directory {conf.path_output} already exists. Skipping.")
        return

        # Run the bias correction for each method
    run_bias_correction(conf, method)


def assess_all():
    # Load the configuration
    conf = Config().get()
    models = conf.RCMs
    methods = conf.bc_methods
    path_inputs = conf.path_inputs
    path_output = conf.path_output

    # Iterate over each model and method
    for model in models:
        for method in methods:
            print(f"Running bias correction for {model} using {method}")

            # Adjust the input and output paths for the current model
            conf.path_inputs = [
                f'{path_inputs[0]}/{model}',
                f'{path_inputs[1]}/{model}'
            ]
            conf.path_output = f'{path_output}/{method}/{model}'

            # If the output directory exists, skip the setting
            if Path(conf.path_output).exists():
                print(f"Output directory {conf.path_output} already exists. Skipping.")
                continue

            # Run the bias correction for each method
            run_bias_correction(conf, method)

    print("Bias correction completed for all models and methods.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        assess_all()
    else:
        assess_single(int(sys.argv[1]))
