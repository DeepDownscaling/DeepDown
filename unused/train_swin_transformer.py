### Using UNetSwinTransformer:  It aims to address the limitations of traditional Vision Transformers 
### by introducing a hierarchical architecture and incorporating shifted windows for local context modeling.
### Builds upon: https://github.com/microsoft/Swin-Transformer

import warnings
import torch.utils.data as data

# Utils
from .utils.data_loader import *
from .utils.utils_loss import *
from .utils.helpers import *
from .utils.data_generators import *
from .models.SUNet import *


warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="divide by zero encountered in divide")

print_cuda_availability()


def main():
    # Load target data
    target = load_target_data(DATE_START, DATE_END, path_targets)
    # Extract the axes of the final target domain based on temperature 
    x_axis = target.TabsD.x
    y_axis = target.TabsD.y

    input_data = load_input_data(DATE_START, DATE_END, PATH_DEM, INPUT_VARIABLES,
                                 INPUT_PATHS,
                                 LEVELS, RESOL_LOW, x_axis, y_axis)

    if DO_CROP:
        input_data = input_data.sel(x=slice(min(CROP_X), max(CROP_X)),
                                    y=slice(max(CROP_Y), min(CROP_Y)))
        target = target.sel(x=slice(min(CROP_X), max(CROP_X)),
                            y=slice(max(CROP_Y), min(CROP_Y)))

    # Split the data (small data for testing purposes)
    x_train = input_data.sel(time=slice('1999', '2011'))
    x_valid = input_data.sel(time=slice('2012', '2015'))
    x_test = input_data.sel(time=slice('2016', '2021'))

    y_train = target.sel(time=slice('1999', '2011'))
    y_valid = target.sel(time=slice('2012', '2015'))
    y_test = target.sel(time=slice('2006', '2011'))

    # Select the variables to use as input and output
    input_vars = {'topo': None, 'tp': None, 't': LEVELS}
    output_vars = ['RhiresD', 'TabsD']  # ['RhiresD', 'TabsD', 'TmaxD', 'TminD']

    training_set = DataGenerator(x_train, y_train, input_vars, output_vars)
    loader_train = torch.utils.data.DataLoader(training_set, batch_size=32)

    # Validation
    valid_set = DataGenerator(x_valid, y_valid, input_vars, output_vars, shuffle=False,
                              mean=training_set.x_mean, std=training_set.x_std)
    loader_val = torch.utils.data.DataLoader(valid_set, batch_size=32)

    # Test
    test_set = DataGenerator(x_test, y_test, input_vars, output_vars, shuffle=False,
                             mean=training_set.x_mean, std=training_set.x_std)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=32)

    torch.cuda.empty_cache()

    model = SUNet(img_size=80, patch_size=5, in_chans=4, out_chans=2, out_size=80,
                  embed_dim=96, depths=[8, 8, 8, 8],
                  num_heads=[8, 8, 8, 8],
                  window_size=20, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                  use_checkpoint=False, final_upsample="Dual up-sample")
    model = model.to(device)

    criterion = nn.MSELoss()
    lr = 0.0001
    # optimizer = optim.SGD(model_with_custom_head.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    num_epochs = 50
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0
        train_predictions = []

        for batch_idx, (inputs, targets) in enumerate(loader_train):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            train_predictions.append(outputs.detach().cpu().numpy())

        train_loss = running_train_loss / len(loader_train)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_predictions = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader_val):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                val_predictions.append(outputs.detach().cpu().numpy())

        val_loss = running_val_loss / len(loader_val)
        val_losses.append(val_loss)

        # torch.save(model.cpu().state_dict(), 'SUNet_checkpoint_adversWP_1e-1.pt')
        torch.save(model.state_dict(), 'SUNet_checkpoint_adversWP_1e-1.pt')


if __name__ == "__main__":
    main()
