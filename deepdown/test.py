import argparse
import warnings
from torch.utils.data import Dataset, DataLoader

from .utils.data_loader import *
from .utils.utils_plot import *
from .utils.utils_loss import *
from .utils.helpers import *
from .utils.data_generators import *
from .models.SRGAN import *
from .train_srgan import *

# Try dask.distributed and see if the performance improves...
from dask.distributed import Client

# c = Client(n_workers=os.cpu_count()-2, threads_per_worker=1)

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="divide by zero encountered in divide")

argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")


def main(config):
    # Paths
    PATH_DEM = config['PATH_DEM']
    PATH_ERA5_025 = config['PATH_ERA5_025']  # Original ERA5 0.25°
    PATH_ERA5_100 = config['PATH_ERA5_100']  # ERA5 1°
    PATH_MCH = config['PATH_MeteoSwiss']

    NUM_CHANNELS_IN = config['NUM_CHANNELS_IN']
    NUM_CHANNELS_OUT = config['NUM_CHANNELS_OUT']
    lr = config['lr']

    # Data options
    DATE_START = '1999-01-01'  # '1979-01-01'
    DATE_END = '2021-12-31'
    YY_TRAIN = [1999, 2015]  # [1979, 2015]
    YY_TEST = [2016, 2021]
    LEVELS = [850, 1000]  # [300, 500, 700, 850, 1000]  # Available with CORDEX-CMIP6
    RESOL_LOW = 0.25  # degrees
    INPUT_VARIABLES = ['tp', 't']
    INPUT_PATHS = [PATH_ERA5_025 + '/precipitation', PATH_ERA5_025 + '/temperature']
    DUMP_DATA_TO_PICKLE = True

    # Crop on a smaller region
    DO_CROP = True
    # I reduce the area of crop now, to avoid NA
    CROP_X = [2700000, 2760000]  # with NAN: [2720000, 2770000]
    CROP_Y = [1190000, 1260000]  # with NAN: [1290000, 1320000]

    # Hyperparameters
    BATCH_SIZE = 32
    # Load target data
    target = load_target_data(DATE_START, DATE_END, PATH_MCH)

    # Extract the axes of the final target domain based on temperature 
    x_axis = target.TabsD.x
    y_axis = target.TabsD.y
    print('loading input data')

    input_data = load_input_data(DATE_START, DATE_END, PATH_DEM, INPUT_VARIABLES,
                                 INPUT_PATHS,
                                 LEVELS, RESOL_LOW, x_axis, y_axis)

    if DO_CROP:
        input_data = input_data.sel(x=slice(min(CROP_X), max(CROP_X)),
                                    y=slice(max(CROP_Y), min(CROP_Y)))
        target = target.sel(x=slice(min(CROP_X), max(CROP_X)),
                            y=slice(max(CROP_Y), min(CROP_Y)))

    # Split the data
    x_train = input_data.sel(time=slice('1999', '2011'))
    x_valid = input_data.sel(time=slice('2012', '2015'))
    x_test = input_data.sel(time=slice('2016', '2021'))

    y_train = target.sel(time=slice('1999', '2011'))
    y_valid = target.sel(time=slice('2012', '2005'))
    y_test = target.sel(time=slice('2006', '2011'))

    # Select the variables to use as input and output
    input_vars = {'topo': None, 'tp': None, 't': LEVELS}
    output_vars = ['RhiresD', 'TabsD']  # ['RhiresD', 'TabsD', 'TmaxD', 'TminD']

    training_set = DataGenerator(x_train, y_train, input_vars, output_vars)
    loader_train = torch.utils.data.DataLoader(training_set, batch_size=32)

    # Validation
    valid_set = DataGenerator(x_valid, y_valid, input_vars, output_vars, shuffle=False,
                              mean=training_set.mean, std=training_set.std)
    loader_val = torch.utils.data.DataLoader(valid_set, batch_size=32)

    # Test
    test_set = DataGenerator(x_test, y_test, input_vars, output_vars, shuffle=False,
                             mean=training_set.mean, std=training_set.std)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Check to make sure the range on the input and output images is correct, and they're the correct shape
    testx, testy = training_set.__getitem__(3)
    print("x shape: ", testx.shape)
    print("y shape: ", testy.shape)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = argParser.parse_args()

    config = read_config(args.config_file)

    print_config(config)

    main(config)
