import yaml
import subprocess
import os

# Load the original YAML configuration
config_path = '../config.yaml'  # Path to your config file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# List of models in RCM
models = config['RCM']
method = config['bias_correction_method']

# Iterate over each model and update path_inputs accordingly
for model in models:
    # Update the path_inputs for the current model
    config['path_inputs'] = [
        f'/storage/workspaces/giub_hydro/hydro/data/RCM/regular-g0d11/precipitation/{model}',
        f'/storage/workspaces/giub_hydro/hydro/data/RCM/regular-g0d11/temperature/{model}'
    ]
    config['path_output'] = f'/storage/workspaces/giub_hydro/hydro/data/tmp2/{method}/{model}'
    
    
    # Create a temporary config file with the updated path_inputs
    temp_config_path = f'temp_config_{model}.yaml'
    with open(temp_config_path, 'w') as temp_file:
        yaml.dump(config, temp_file)

    print("running bias for", model)

    # Run the bias_correction.py script with the updated config
    command = f'python deepdown/bias_correction_sbck.py --config {temp_config_path}'
    subprocess.run(command, shell=True)

    # Optionally, remove the temporary config file after running the script
    os.remove(temp_config_path)
