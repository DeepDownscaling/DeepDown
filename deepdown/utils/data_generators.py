import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr


# Create data generator in pytorch - Adapted from the keras class
class DataGenerator(Dataset):
    def __init__(self, dx, dy, input_vars, output_vars, shuffle=True, load=True,
                 mean=None, std=None, tp_log=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            dx: Dataset containing all input variables
            dy: Dataset containing the data to predict
            input_vars: Dictionary of the form {'var': level}. Use None for level if data is of single level
            output_vars: List of variables to be predicted
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
            tp_log: Log transformation for precipitation. If None, no transformation is applied.
        """

        self.dx = dx
        self.dy = dy
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.shuffle = shuffle

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])

        for var, levels in input_vars.items():
            # Variables transformation
            if var == 'tp' and tp_log:
                data = self.log_trans(dx[var], tp_log)

            # Handle dimensions
            if var == 'topo':
                data.append(dx[var].expand_dims(
                    {'level': generic_level, 'time': dx.time}, (1, 0)
                ))
            elif levels is None:
                data.append(dx[var].expand_dims({'level': generic_level}, 1))
            else:
                data.append(dx[var].sel(level=levels))

        # In PyTorch we must transpose (C,H,W,B)
        self.data = xr.concat(data, 'level').transpose('level', 'y', 'x', 'time')

        # Normalize 
        self.mean = self.data.mean(
            ('time', 'y', 'x')).compute() if mean is None else mean
        self.std = self.data.mean(('time', 'y', 'x')).compute() if std is None else std
        self.data = (self.data - self.mean) / self.std

        # Get indices of samples
        self.n_samples = self.data.shape[3]
        self.on_epoch_end()

        # Prepare the target
        self.dy = [self.dy[var] for var in self.output_vars]

        # Concatenate the DataArray objects along a new dimension
        self.dy = xr.concat(self.dy, dim='level').transpose('level', 'y', 'x', 'time')

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load:
            print('Loading data into RAM')
            self.data.load()
            self.dy.load()

    def __len__(self):
        'Denotes the number of samples'
        return self.n_samples

    def __getitem__(self, idx):
        'Loads and returns a sample from the dataset'
        idxs = self.idxs[idx]
        X = (torch.Tensor(self.data.isel(time=idxs).values))
        y = (torch.Tensor(self.dy.isel(time=idxs).values))

        if y.ndim == 2:
            # Expand dimensions
            y = torch.unsqueeze(y, dim=0)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

    def log_trans(x, e):
        return np.log(x + e) - np.log(e)

    def log_retrans(x, e):
        return np.exp(x + np.log(e)) - e
