import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr


# Create data generator in pytorch - Adapted from the keras class
class DataGenerator(Dataset):
    def __init__(self, inputs, outputs, input_vars, output_vars, do_crop=True,
                 crop_x=None, crop_y=None, shuffle=True, x_mean=None, x_std=None,
                 standardize_y=False, y_mean=None, y_std=None, tp_log=None):
        """
        Data generator. Template from:
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        Parameters
        ----------
        inputs: xarray Dataset
            Input data.
        outputs: xarray Dataset
            Data to predict.
        input_vars: dict
            Dictionary of input variables. Keys are variable names and values are
            the pressure levels. Use None for variables on a single level.
        output_vars: list
            Dictionary of output variables. Keys are variable names and values are
            the pressure levels. Use None for variables on a single level.
        do_crop: bool
            If True, crops the input and output data.
        crop_x: list
            List with the minimum and maximum x coordinates to crop.
        crop_y: list
            List with the minimum and maximum y coordinates to crop.
        shuffle: bool
            If True, the data is shuffled.
        x_mean: xarray Dataset
            Mean to subtract from input data for normalization.
            If None, it is computed from the data.
        x_std: xarray Dataset
            Standard deviation to subtract from input data for normalization.
            If None, it is computed from the data.
        standardize_y: bool
            If True, standardizes the target data.
        y_mean: xarray Dataset
            Mean to subtract from target data for normalization.
            If None, it is computed from the data.
        y_std: xarray Dataset
            Standard deviation to subtract from target data for normalization.
            If None, it is computed from the data.
        tp_log: ?float
            If not None, applies a log transformation to the variable 'tp' using the
            given value as a threshold.
        """

        self.shuffle = shuffle
        self.idxs = None

        inputs = self.rev_lat(inputs)
        outputs = self.rev_lat(outputs)

        # Crop option
        if do_crop:
            inputs = inputs.sel(x=slice(min(crop_x), max(crop_x)),
                                y=slice(max(crop_y), min(crop_y)))
            outputs = outputs.sel(x=slice(min(crop_x), max(crop_x)),
                                  y=slice(max(crop_y), min(crop_y)))

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])

        for var, levels in input_vars.items():
            # Variables transformation
            if var == 'tp' and tp_log:
                data = self.log_trans(inputs[var], tp_log)

            # Handle dimensions
            if var == 'topo':
                data.append(inputs[var].expand_dims(
                    {'level': generic_level, 'time': inputs.time}, (1, 0)
                ))
            elif levels is None or isinstance(levels, str) and levels == 'None':
                data.append(inputs[var].expand_dims({'level': generic_level}, 1))
            else:
                data.append(inputs[var].sel(level=levels))

        # In PyTorch we must transpose (B,C,H,W)
        self.x = xr.concat(data, 'level').transpose('time', 'level', 'y', 'x')

        # Standardize input data
        print('Computing/assigning mean and std...')
        self.x_mean = self.x.mean(
            ('time', 'y', 'x')).compute() if x_mean is None else x_mean
        self.x_std = self.x.std(
            ('time', 'y', 'x')).compute() if x_std is None else x_std
        self.x = (self.x - self.x_mean) / self.x_std

        # Prepare the target data
        self.y = [outputs[var] for var in output_vars]
        self.y = xr.concat(self.y, 'level').transpose('time', 'level', 'y', 'x')

        # Standardize outputs
        if standardize_y:
            self.y_mean = self.y.mean(
                ('time', 'y', 'x')).compute() if y_mean is None else y_mean
            self.y_std = self.y.std(
                ('time', 'y', 'x')).compute() if y_std is None else y_std
            self.y = (self.y - self.y_mean) / self.y_std

        # Get indices of samples
        self.n_samples = len(self.x)
        print("Number of samples", self.n_samples)
        self.on_epoch_end()

    @staticmethod
    def rev_lat(data):
        if data['y'][0].values < data['y'][1].values:
            # If the first latitude is less than the second, reverse the order
            data['y'] = data['y'][data['y'].argsort()[::-1]]
        return data

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
