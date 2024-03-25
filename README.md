# DeepDown

Downscaling climate variables, with the focus on precipitation and temperature, 
over Switzerland using reanalysis products (low-to-high resolution).

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Configuration

The configuration is done through config files and arguments (to come).
There is a configuration file `config.default.yaml` that contains the the default parameters values.
An additional user-based config file is required to run the code and should contain at a minimum the necessary path (an example of such a file is provided in `config.yaml.example`).
The options are enforced in the following order:
1. Starting with the content of the `config.default.yaml` file
2. Overridden by the content of the `config.yaml` file
3. Overridden by the arguments (to be implemented).


## Note

The folder `unused` contains code that is not used anymore. It is kept for reference 
and potential future use.
