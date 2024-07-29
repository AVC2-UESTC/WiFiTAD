import torch
import torch.nn as nn
import numpy as np

# ========================== #
# He Initialization Function #
# ========================== #

def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif (isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d)) and m.affine:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# ====================== #
# Fiducial Starter Layer #
# ====================== #

class ResidualStarter(nn.Module):
    def __init__(self, in_samples, out_samples, in_channels, out_channels, kernel_size, *, verbose=False, activation=None):
        super(ResidualStarter, self).__init__()

        if activation is None:
            activation = nn.ReLU

        if in_samples < out_samples:
            raise ValueError("Input samples must be greater than or equal to output samples")
        if in_samples % out_samples != 0:
            raise ValueError("Input channels must be divisible by output channels")
        downsample = in_samples // out_samples

        self._conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=downsample, padding=(kernel_size-1)//2, bias=False)
        self._bn = nn.BatchNorm1d(out_channels)
        self._activation = activation()

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = self._activation(x)
        return x

# ================= #
# ResNet Components #
# ================= #

class ResidualLayer(nn.Module):
    def __init__(self, in_samples, out_samples, in_channels, out_channels, kernel_size, *, dropout=-1, verbose=False, activation=None):
        super(ResidualLayer, self).__init__()
        if verbose:
            print(f"Residual Unit: {in_samples} -> {out_samples} samples, {in_channels} -> {out_channels} channels, kernel size {kernel_size}")

        if activation is None:
            activation = nn.ReLU

        if in_samples < out_samples:
            raise ValueError("Input samples must be greater than or equal to output samples")
        if in_samples % out_samples != 0:
            raise ValueError("Input channels must be divisible by output channels")
        downsample = in_samples // out_samples

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        padding = (kernel_size - 1) // 2

        # Skip connection
        self._skip_pooling = nn.MaxPool1d(downsample, stride=downsample) if downsample > 1 else nn.Identity()
        self._skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

        self._conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=1, bias=False)

        # Batch norm and ReLU
        self._bn_1 = nn.BatchNorm1d(out_channels)
        self._activation_1 = activation()

        # Second convolutional Layer
        self._conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=downsample, bias=False, padding=padding)

        # Batch norm and ReLU
        self._bn_2 = nn.BatchNorm1d(out_channels)
        self._activation_2 = activation()

        if dropout > 0:
            self._dropout_layer = nn.Dropout(dropout)
        else:
            self._dropout_layer = nn.Identity()

    def forward(self, x, y):
        # Main line
        x = self._conv_1(x)
        x = self._bn_1(self._activation_1(x))
        x = self._dropout_layer(x)
        x = self._conv_2(x)

        # Skip line
        y = self._skip_conv(y)
        y = self._skip_pooling(y)

        # Adding
        added = x + y
        x, y = added, added

        # Final batch norm and ReLU
        x = self._bn_2(self._activation_2(x))
        x = self._dropout_layer(x)

        return x, y

class ResidualBlock(nn.Module):
    def __init__(self, in_samples, out_samples, in_channels, out_channels, kernel_size, num_layers, *, dropout=-1, verbose=False, activation=None):
        super(ResidualBlock, self).__init__()
        if verbose:
            print(f"Residual Block: {in_samples} -> {out_samples} samples, {in_channels} -> {out_channels} channels, kernel size {kernel_size}, {num_layers} layers:")

        self._layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels
            in_sm = in_samples if i == 0 else out_samples
            out_sm = out_samples
            if verbose:
                print(f"Layer {i+1}/{num_layers}: ", end="")
            self._layers.append(ResidualLayer(in_sm, out_sm, in_ch, out_ch, kernel_size, dropout=dropout, verbose=verbose, activation=activation))

    def forward(self, x, y):
        for layer in self._layers:
            x, y = layer(x, y)
        return x, y


# ===================== #
# BottleNeck Components #
# ===================== #

class BottleNeckResidualLayer(nn.Module):
    def __init__(self, in_samples, out_samples, in_channels, mid_channels, out_channels, kernel_size, *, dropout=-1, verbose=False, activation=None):
        super(BottleNeckResidualLayer, self).__init__()
        if verbose:
            print(f"Bottle Neck Residual Unit: {in_samples} -> {out_samples} samples, {in_channels} -> {mid_channels} -> {out_channels} channels, kernel size {kernel_size}")

        if activation is None:
            activation = nn.ReLU

        if in_samples < out_samples:
            raise ValueError("Input samples must be greater than or equal to output samples")
        if in_samples % out_samples != 0:
            raise ValueError("Input channels must be divisible by output channels")
        downsample = in_samples // out_samples

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        padding = (kernel_size - 1) // 2

        # Skip connection
        self._skip_pooling = nn.MaxPool1d(downsample, stride=downsample) if downsample > 1 else nn.Identity()
        self._skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

        # First convolutional Layer
        self._conv_1 = nn.Conv1d(in_channels, mid_channels, 1, padding=0, stride=1, bias=False)
        self._bn_1 = nn.BatchNorm1d(mid_channels)
        self._activation_1 = activation()

        # Second convolutional Layer
        self._conv_2 = nn.Conv1d(mid_channels, mid_channels, kernel_size, stride=downsample, bias=False, padding=padding)
        self._bn_2 = nn.BatchNorm1d(mid_channels)
        self._activation_2 = activation()

        # Third convolutional Layer
        self._conv_3 = nn.Conv1d(mid_channels, out_channels, 1, padding=0, stride=1, bias=False)
        self._bn_3 = nn.BatchNorm1d(out_channels)
        self._activation_3 = activation()

        if dropout > 0:
            self._dropout_layer = nn.Dropout(dropout)
        else:
            self._dropout_layer = nn.Identity()

    def forward(self, x, y):
        # Main line
        # 1
        x = self._conv_1(x)
        x = self._bn_1(self._activation_1(x))
        x = self._dropout_layer(x)
        # 2
        x = self._conv_2(x)
        x = self._bn_2(self._activation_2(x))
        x = self._dropout_layer(x)
        # 3
        x = self._conv_3(x)

        # Skip line
        y = self._skip_conv(y)
        y = self._skip_pooling(y)

        # Adding
        added = x + y
        x, y = added, added

        # Final batch norm and ReLU (preactivation)
        x = self._bn_3(self._activation_3(x))
        x = self._dropout_layer(x)

        return x, y

class BottleNeckResidualBlock(nn.Module):
    def __init__(self, in_samples, out_samples, in_channels, mid_channels, out_channels, kernel_size, num_layers, *, dropout=-1, verbose=True, activation=None):
        super(BottleNeckResidualBlock, self).__init__()
        if verbose:
            print(f"Bottle Neck Residual Block: {in_samples} -> {out_samples} samples, {in_channels} -> {mid_channels} -> {out_channels} channels, kernel size {kernel_size}, {num_layers} layers:")

        self._layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels
            in_sm = in_samples if i == 0 else out_samples
            out_sm = out_samples
            if verbose:
                print(f"Layer {i+1}/{num_layers}: ", end="")
            self._layers.append(BottleNeckResidualLayer(in_sm, out_sm, in_ch, mid_channels, out_ch, kernel_size, dropout=dropout, verbose=verbose, activation=activation))

    def forward(self, x, y):
        for layer in self._layers:
            x, y = layer(x, y)
        return x, y

# ==================== #
# Network Base Classes #
# ==================== #

class ResNet(nn.Module):
    def __init__(self,
                 in_samples, in_channels,
                 starter_channels, starter_kernel,
                 blocks_channels_kernels_layers_list,
                 *,
                 activation=None, verbose=False,
                 half_start=True,
                 dropout=-1):
        super(ResNet, self).__init__()

        # Check the number of samples is appropriate
        samples_test = in_samples
        num_halves_needed = len(blocks_channels_kernels_layers_list) + 1 if half_start else len(blocks_channels_kernels_layers_list)
        for _ in range(num_halves_needed):
            if samples_test % 2 != 0:
                raise ValueError("Number of samples must be divisible by 2")
            if samples_test == 1:
                raise ValueError("Number of samples must be at least 2")
            samples_test //= 2

        # Build the starter convoltion layer
        current_samples, current_channels = in_samples, in_channels
        if half_start:
            self.starter = ResidualStarter(current_samples, current_samples//2, current_channels, starter_channels, starter_kernel, verbose=verbose)
            current_samples, current_channels = current_samples//2, starter_channels
        else:
            self.starter = ResidualStarter(current_samples, current_samples, current_channels, starter_channels, starter_kernel, verbose=verbose)
            current_samples, current_channels = in_samples, starter_channels

        # Build the residual blocks
        self.blocks = nn.ModuleList()
        for channels, kernel, layers in blocks_channels_kernels_layers_list:
            self.blocks.append(ResidualBlock(current_samples, current_samples, current_channels, channels, kernel, layers, dropout=dropout, verbose=verbose, activation=activation))
            current_samples, current_channels = current_samples, channels

        # Build the ender
        self.ender = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        self.apply(initialize_weights)

        # Get useful parameters #
        self.num_layers = sum([2*layers for _, _, layers in blocks_channels_kernels_layers_list]) + 1
        # Note that the layer here contains two convolutions for each skip connection. This is why the number of layers is doubled
        # Also note that we don't have a FC layer at the end, so our convention is -1 compared to literature
        self.final_channels = blocks_channels_kernels_layers_list[-1][0]
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.num_blocks = len(blocks_channels_kernels_layers_list)
        self.num_flops = 0


    def forward(self, x):
        x = self.starter(x)
        y = x
        for layer in self.blocks:
            x, y = layer(x, y)
        # x = self.ender(x)
        return x

class BottleNeckResNet(nn.Module):
    def __init__(self,
                 in_samples, in_channels,
                 starter_channels, starter_kernel,
                 blocks_midchannels_endchannels_kernels_layers_list,
                 *,
                 activation=None, verbose=False,
                 half_start=True,
                 dropout=-1):
        super(BottleNeckResNet, self).__init__()

        # Check the number of samples is appropriate
        samples_test = in_samples
        num_halves_needed = len(blocks_midchannels_endchannels_kernels_layers_list) + 1 if half_start else len(blocks_midchannels_endchannels_kernels_layers_list)
        for _ in range(num_halves_needed):
            if samples_test % 2 != 0:
                raise ValueError("Number of samples must be divisible by 2")
            if samples_test == 1:
                raise ValueError("Number of samples must be at least 2")
            samples_test //= 2

        # Build the starter convoltion layer
        current_samples, current_channels = in_samples, in_channels
        if half_start:
            self.starter = ResidualStarter(current_samples, in_samples//2, current_channels, starter_channels, starter_kernel, verbose=verbose)
            current_samples, current_channels = in_samples//2, starter_channels
        else:
            self.starter = ResidualStarter(current_samples, current_samples, current_channels, starter_channels, starter_kernel, verbose=verbose)
            current_samples, current_channels = in_samples, starter_channels

        # Build the residual blocks
        self.blocks = nn.ModuleList()
        for mid_channels, end_channels, kernel, layers in blocks_midchannels_endchannels_kernels_layers_list:
            self.blocks.append(BottleNeckResidualBlock(current_samples, current_samples//2, current_channels, mid_channels, end_channels, kernel, layers, dropout=dropout, verbose=verbose, activation=activation))
            current_samples, current_channels = current_samples//2, end_channels

        # Build the ender
        self.ender = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        self.apply(initialize_weights)

        # Get useful parameters #

        self.num_layers = sum([3*layers for _, _, _, layers in blocks_midchannels_endchannels_kernels_layers_list]) + 1
        # Note that the layer here contains two convolutions for each skip connection. This is why the number of layers is doubled
        # Also note that we don't have a FC layer at the end, so our convention is -1 compared to literature
        self.final_channels = blocks_midchannels_endchannels_kernels_layers_list[-1][1]
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.num_blocks = len(blocks_midchannels_endchannels_kernels_layers_list)

    def forward(self, x):
        x = self.starter(x)
        y = x
        for layer in self.blocks:
            x, y = layer(x, y)
        # x = self.ender(x)
        return x


# =============== #
# Concrete Models #
# =============== #

class ResNet17(ResNet):
    def __init__(self,
                in_samples,
                in_channels=1, starter_channels=64, end_channels=160,
                starter_kernel=15, mid_kernel=3, *,
                activation=None, logchannels=True,
                verbose=True, half_start=True,
                dropout=-1):

        if activation is None:
            activation = nn.ReLU

        self.save_str = f"ResNet17_{in_samples}_{in_channels}_{starter_channels}_{end_channels}_{starter_kernel}_{mid_kernel}_{activation.__name__}_{'log' if logchannels else 'lin'}_{'halved' if half_start else 'not-halved'}"

        if logchannels:
            channels = np.round(np.geomspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)
        else:
            channels = np.round(np.linspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)

        blocks_channels_kernels_layers_list = [(channels[0], mid_kernel, 2), (channels[1], mid_kernel, 2), (channels[2], mid_kernel, 2), (channels[3], mid_kernel, 2)]


        super(ResNet17, self).__init__(
            in_samples, in_channels,
            starter_channels, starter_kernel,
            blocks_channels_kernels_layers_list,
            activation=activation,
            verbose=verbose,
            half_start=half_start,
            dropout=dropout
        )

    def get_save_string(self):
        return self.save_str

class ResNet33(ResNet):
    def __init__(self,
                in_samples,
                in_channels=1, starter_channels=64, end_channels=160,
                starter_kernel=15, mid_kernel=3, *,
                activation=None, logchannels=True,
                verbose=True, half_start=True,
                dropout=-1):

        if activation is None:
            activation = nn.ReLU

        self.save_str = f"ResNet33_{in_samples}_{in_channels}_{starter_channels}_{end_channels}_{starter_kernel}_{mid_kernel}_{activation.__name__}_{'log' if logchannels else 'lin'}_{'halved' if half_start else 'not-halved'}"

        if logchannels:
            channels = np.round(np.geomspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)
        else:
            channels = np.round(np.linspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)

        blocks_channels_kernels_layers_list = [(channels[0], mid_kernel, 3), (channels[1], mid_kernel, 4), (channels[2], mid_kernel, 6), (channels[3], mid_kernel, 3)]

        super(ResNet33, self).__init__(
            in_samples, in_channels,
            starter_channels, starter_kernel,
            blocks_channels_kernels_layers_list,
            activation=activation,
            verbose=verbose,
            half_start=half_start,
            dropout=dropout
        )

    def get_save_string(self):
        return self.save_str

class ResNet49(BottleNeckResNet):
    def __init__(self,
                in_samples,
                in_channels=1, starter_channels=64, end_channels=160,
                starter_kernel=15, mid_kernel=3, *,
                activation=None, logchannels=True,
                verbose=True, half_start=True,
                dropout=-1):

        if activation is None:
            activation = nn.ReLU

        self.save_str = f"ResNet49_{in_samples}_{in_channels}_{starter_channels}_{end_channels}_{starter_kernel}_{mid_kernel}_{activation.__name__}_{'log' if logchannels else 'lin'}_{'halved' if half_start else 'not-halved'}"

        if logchannels:
            channels = np.round(np.geomspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)
        else:
            channels = np.round(np.linspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)

        blocks_midchannels_endchannels_kernels_layers_list = [(channels[0]//4, channels[0], mid_kernel, 3), (channels[1]//4, channels[1], mid_kernel, 4), (channels[2]//4, channels[2], mid_kernel, 6), (channels[3]//4, channels[3], mid_kernel, 3)]

        super(ResNet49, self).__init__(
            in_samples, in_channels,
            starter_channels, starter_kernel,
            blocks_midchannels_endchannels_kernels_layers_list,
            activation=activation,
            verbose=verbose,
            half_start=half_start,
            dropout=dropout
        )

    def get_save_string(self):
        return self.save_str

class ResNet100(BottleNeckResNet):
    def __init__(self,
                in_samples,
                in_channels=1, starter_channels=64, end_channels=160,
                starter_kernel=15, mid_kernel=3, *,
                activation=None, logchannels=True,
                verbose=True, half_start=True,
                dropout=-1):

        if activation is None:
            activation = nn.ReLU

        self.save_str = f"ResNet100_{in_samples}_{in_channels}_{starter_channels}_{end_channels}_{starter_kernel}_{mid_kernel}_{activation.__name__}_{'log' if logchannels else 'lin'}_{'halved' if half_start else 'not-halved'}"

        if logchannels:
            channels = np.round(np.geomspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)
        else:
            channels = np.round(np.linspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)

        blocks_midchannels_endchannels_kernels_layers_list = [(channels[0]//4, channels[0], mid_kernel, 3), (channels[1]//4, channels[1], mid_kernel, 4), (channels[2]//4, channels[2], mid_kernel, 23), (channels[3]//4, channels[3], mid_kernel, 3)]

        super(ResNet100, self).__init__(
            in_samples, in_channels,
            starter_channels, starter_kernel,
            blocks_midchannels_endchannels_kernels_layers_list,
            activation=activation,
            verbose=verbose,
            half_start=half_start,
            dropout=dropout
        )

    def get_save_string(self):
        return self.save_str

class ResNet151(BottleNeckResNet):
    def __init__(self,
                in_samples,
                in_channels=1, starter_channels=64, end_channels=160,
                starter_kernel=15, mid_kernel=3, *,
                activation=None, logchannels=True,
                verbose=True, half_start=True,
                dropout=-1):

        if activation is None:
            activation = nn.ReLU

        self.save_str = f"ResNet151_{in_samples}_{in_channels}_{starter_channels}_{end_channels}_{starter_kernel}_{mid_kernel}_{activation.__name__}_{'log' if logchannels else 'lin'}_{'halved' if half_start else 'not-halved'}"

        if logchannels:
            channels = np.round(np.geomspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)
        else:
            channels = np.round(np.linspace(starter_channels, end_channels, 4, endpoint=True)).astype(int)

        blocks_midchannels_endchannels_kernels_layers_list = [(channels[0]//4, channels[0], mid_kernel, 3), (channels[1]//4, channels[1], mid_kernel, 8), (channels[2]//4, channels[2], mid_kernel, 36), (channels[3]//4, channels[3], mid_kernel, 3)]

        super(ResNet151, self).__init__(
            in_samples, in_channels,
            starter_channels, starter_kernel,
            blocks_midchannels_endchannels_kernels_layers_list,
            activation=activation,
            verbose=verbose,
            half_start=half_start,
            dropout=dropout
        )

    def get_save_string(self):
        return self.save_str