import torch.nn as nn
import torch.nn.functional as F

# Adapted from: https://github.com/mantariksh/231n_downscaling/blob/master/SRGAN.ipynb
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(num_channels)
        )

    def forward(self, x):
        return x + self.layers(x)


        
class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, num_channels, input_size, output_channels =4, num_res_blocks=16, scale_factor=1):
        super().__init__()
        
        self.num_res_blocks = num_res_blocks
        self.input_size = input_size
        self.output = output_channels
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(num_channels, 64, kernel_size=9, stride=1, padding=0),
            nn.PReLU()
        )
        
        self.resBlocks = nn.ModuleList([ResidualBlock(64) for i in range(self.num_res_blocks)])

        self.post_resid_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64)
        )

        self.upscale_blocks = nn.ModuleList()
        for _ in range(int(math.log(scale_factor, 2))):
            self.upscale_blocks.append(UpscaleBlock(64, 64 * 4, scale_factor=2))

        self.conv_prelu = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.PReLU()
        )
    
        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(64, self.output, 9, stride=1, padding=0)
        )

    def forward(self, x):
        initial_conv_out = self.initial_conv(x)
                
        res_block_out = self.resBlocks[0](initial_conv_out)
        for i in range(1, self.num_res_blocks):
            res_block_out = self.resBlocks[i](res_block_out)

        post_resid_conv_out = self.post_resid_conv(res_block_out) + initial_conv_out

        upscale_out = post_resid_conv_out
        for block in self.upscale_blocks:
            upscale_out = block(upscale_out)

        conv_prelu_out = self.conv_prelu(upscale_out)
        final_out = self.final_conv(conv_prelu_out)
        # To get the final desired shape
       # print(final_out.shape)
        final_out = F.interpolate(final_out, size=self.input_size, mode='bicubic', align_corners=True)

        return final_out


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Discriminator(nn.Module):
    def __init__(self, num_channels, H, W):
        super().__init__()
        
        self.layers = nn.Sequential( 
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            Flatten(),  
            nn.Linear(512 * int(np.ceil(H/16)) * int(np.ceil(W/16)), 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            
        )
        
    def forward(self, x):
        return self.layers(x)