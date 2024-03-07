import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr


# Create data generator in pytorch - Adapted from the keras class
class DataGenerator(Dataset):
    def __init__(self, x, y, input_vars, output_vars, shuffle=True, load=False,
                 mean=None, std=None, tp_log=None):
        """
        Data generator. Template from:
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        Parameters
        ----------
        x: xarray Dataset
            Input data.
        y: xarray Dataset
            Data to predict.
        input_vars: dict
            Dictionary of input variables. Keys are variable names and values are
            the pressure levels. Use None for variables on a single level.
        output_vars: list
            Dictionary of output variables. Keys are variable names and values are
            the pressure levels. Use None for variables on a single level.
        shuffle: bool
            If True, the data is shuffled.
        load: bool
            If True, the data is loaded into memory.
        mean: xarray Dataset
            Mean to subtract from data for normalization.
            If None, it is computed from the data.
        std: xarray Dataset
            Standard deviation to subtract from data for normalization.
            If None, it is computed from the data.
        tp_log: float
            If not None, applies a log transformation to the variable 'tp' using the
            given value as a threshold.
        """

        self.shuffle = shuffle
        self.idxs = None

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])

        for var, levels in input_vars.items():
            # Variables transformation
            if var == 'tp' and tp_log:
                data = self.log_trans(x[var], tp_log)

            # Handle dimensions
            if var == 'topo':
                data.append(x[var].expand_dims(
                    {'level': generic_level, 'time': x.time}, (1, 0)
                ))
            elif levels is None:
                data.append(x[var].expand_dims({'level': generic_level}, 1))
            else:
                data.append(x[var].sel(level=levels))

        # In PyTorch we must transpose (C,H,W,B)
        self.x = xr.concat(data, 'level').transpose('level', 'y', 'x', 'time')

        # Normalize
        print('Computing/assigning mean and std...')
        self.mean = self.x.mean(
            ('time', 'y', 'x')).compute() if mean is None else mean
        self.std = self.x.std(
            ('time', 'y', 'x')).compute() if std is None else std
        self.x = (self.x - self.mean) / self.std

        # Get indices of samples
        self.n_samples = self.x.shape[3]
        self.on_epoch_end()

        # Prepare the target
        self.y = [y[var] for var in output_vars]

        # Concatenate the DataArray objects along a new dimension
        self.y = xr.concat(self.y, dim='level').transpose('level', 'y', 'x', 'time')

        if load:
            print('Loading data into RAM...')
            self.x.load()
            self.y.load()

    def __len__(self):
        """Denotes the number of samples"""
        return self.n_samples

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset"""
        idxs = self.idxs[idx]
        X = (torch.Tensor(self.x.isel(time=idxs).values))
        y = (torch.Tensor(self.y.isel(time=idxs).values))

        if y.ndim == 2:
            # Expand dimensions
            y = torch.unsqueeze(y, dim=0)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.idxs = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.idxs)

    @staticmethod
    def log_trans(x, e):
        return np.log(x + e) - np.log(e)

    @staticmethod
    def log_retrans(x, e):
        return np.exp(x + np.log(e)) - e
