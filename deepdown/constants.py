import yaml

# Define paths and constant
with open('../config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Display options
PLOT_DATA_FULL_EXTENT = False
PLOT_DATA_CROPPED = False

PATH_DEM = config['PATH_DEM']
PATH_ERA5_025 = config['PATH_ERA5_025']  # Original ERA5 0.25°
PATH_ERA5_100 = config['PATH_ERA5_100']  # ERA5 1°
PATH_MCH = config['PATH_MeteoSwiss']

NUM_CHANNELS_IN = config['NUM_CHANNELS_IN']
NUM_CHANNELS_OUT = config['NUM_CHANNELS_OUT']
lr = config['lr']

# Data options
DATE_START = config['DATE_START']
DATE_END = config['DATE_END']
YY_TRAIN = config['YY_TRAIN']
YY_TEST = config['YY_TEST']
LEVELS = config['LEVELS']
RESOL_LOW = config['RESOL_LOW']
INPUT_VARIABLES = config['INPUT_VARIABLES']
INPUT_PATHS = [PATH_ERA5_025 + '/precipitation', PATH_ERA5_025 + '/temperature']
DUMP_DATA_TO_PICKLE = config['DUMP_DATA_TO_PICKLE']

NUM_CHANNELS_IN = config['NUM_CHANNELS_IN']
NUM_CHANNELS_OUT = config['NUM_CHANNELS_OUT']
lr = config['lr']
# Crop on a smaller region
DO_CROP = config['DO_CROP']
# I reduce the area of crop now, to avoid NA
CROP_X = [2680000, 2760000]  # with NAN: [2720000, 2770000]
CROP_Y = [1180000, 1260000]  # with NAN: [1290000, 1320000]

device = config['device']
dtype = config['dtype']
# Hyperparameters
BATCH_SIZE = 32
h, w = config['lowres_shape']
