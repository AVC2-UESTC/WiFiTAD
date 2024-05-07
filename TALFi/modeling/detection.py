import torch
import torch.nn as nn
from TALFi.modeling.ku2 import Unit1D

class Tower(nn.Module):
    def __init__(self, out_channels, layer):
        super().__init__()
        
        conf_towers = [] 
        for i in range(layer):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        self.conf_tower = nn.Sequential(*conf_towers)

    def forward(self, x):
        return self.conf_tower(x)


class loc_head_af(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.center = nn.Sequential(
            Unit1D(
                in_channels=out_channels,
                output_channels=out_channels,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(
                in_channels=out_channels,
                output_channels=1,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None)
            )
        self.duration = nn.Sequential(
            Unit1D(
                in_channels=out_channels,
                output_channels=out_channels,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(
                in_channels=out_channels,
                output_channels=1,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None)
            )
        
        self.bin_num = 1
        self.bin = nn.Sequential(
            Unit1D(
                in_channels=out_channels,
                output_channels=out_channels,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(
                in_channels=out_channels,
                output_channels=self.bin_num*2+1,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None)
            )

    def forward(self, x):
        duration = self.duration(x)
        center = self.center(x)
        bin = self.bin(x)

        center_result = sliding_window(center, window_size=2*self.bin_num+1, mode=3)
        # print(center_result.shape, bin.shape)
        center_result = center_result * bin
        # print(center_result.shape)
        # Now, sum along the bin dimension, not mean
        center_result = torch.mean(center_result, dim=1, keepdim=True)

        w_duration = duration.repeat(1,2,1)
        w_duration[:, 0, :] = w_duration[:, 0, :] - center_result
        w_duration[:, 1, :] = w_duration[:, 1, :] + center_result
        return w_duration
    
def sliding_window(tensor, window_size, mode=1, step_size=1):
    # print(111, tensor.shape)
    if mode == 1:
        # Add padding to the end, position is "position + window_size"
        tensor = torch.nn.functional.pad(tensor, (0, window_size-step_size))
        # print(222, tensor.shape)
    elif mode == 2:
        # Add padding to the start, position is "position - window_size"
        tensor = torch.nn.functional.pad(tensor, (window_size-step_size, 0))
    elif mode == 3:
        # Add padding to the start, position is "position - window_size"
        tensor = torch.nn.functional.pad(tensor, ((window_size-1)//2, (window_size-1)//2))
    # print(222, tensor.shape)
    new_matrix = tensor.unfold(2, window_size, step_size).transpose(1, 2).squeeze(-2).permute(0,2,1)
    # print(333, new_matrix.shape)
    return new_matrix


class conf_head_af(nn.Module):
    def __init__(self, out_channels=512, num_classes=8):
        super().__init__()
        self.conf = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

    def forward(self, x):
        x = self.conf(x)
        return x


class loc_head_af2(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.loc = Unit1D(
                in_channels=out_channels,
                output_channels=2,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            )

    def forward(self, x):
        x = self.loc(x)
        return x

class loc_head_af3(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.center = nn.Sequential(
            Unit1D(
                in_channels=out_channels,
                output_channels=out_channels,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(
                in_channels=out_channels,
                output_channels=1,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None)
            )
        self.duration = nn.Sequential(
            Unit1D(
                in_channels=out_channels,
                output_channels=out_channels,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(
                in_channels=out_channels,
                output_channels=1,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None)
            )
    def forward(self, x):
        duration = self.duration(x)
        center_result = self.center(x)
       
        w_duration = duration.repeat(1,2,1)
        w_duration[:, 0, :] = w_duration[:, 0, :] - center_result
        w_duration[:, 1, :] = w_duration[:, 1, :] + center_result
        return w_duration

class detection_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_tower = Tower(512, 3)
        self.conf_tower = Tower(512, 3)
        
        self.loc_head = loc_head_af3()
        self.conf_head = conf_head_af()
        
    def forward(self, x):
        loc_feat = self.loc_tower(x)
        conf_feat = self.conf_tower(x)
        
        loc_feat = self.loc_head(loc_feat)
        conf_feat = self.conf_head(conf_feat)
        
        return loc_feat, conf_feat


# class detection_head(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.taylor = 3
#         self.loc_tower = nn.ModuleList([Tower(512, 3) for _ in range(self.taylor)])
#         self.conf_tower = nn.ModuleList([Tower(512, 3) for _ in range(self.taylor)])
        
#         self.loc_head = nn.ModuleList([loc_head_af2() for _ in range(self.taylor)])
#         self.conf_head = nn.ModuleList([conf_head_af() for _ in range(self.taylor)])
        
#     def forward(self, x):
#         loc_feat = x
#         conf_feat = x
#         loc_feat_taylor = torch.zeros(x.size(0), 2, x.size(2)).to(x.device)  # 初始化为零张量
#         conf_feat_taylor = torch.zeros(x.size(0), 8, x.size(2)).to(x.device)
#         for i in range(0, self.taylor):
#             loc_feat = self.loc_tower[i](loc_feat)
#             conf_feat = self.conf_tower[i](conf_feat)
#             loc_feat_taylor += torch.pow(self.loc_head[i](loc_feat), i+1)/torch.prod(torch.arange(1, i+1).float())
#             conf_feat_taylor += self.conf_head[i](conf_feat)



#         # loc_feat_taylor = self.loc_head[0](loc_feat)
#         # conf_feat_taylor = self.conf_head[0](conf_feat)
#         # for i in range(1, 3):
#         #     if i == 1:
#         #         loc_feat_taylor -= torch.pow(self.loc_head[i](loc_feat), 3)/torch.prod(torch.arange(1, 4).float())
#         #     if i == 2:
#         #         loc_feat_taylor += torch.pow(self.loc_head[i](loc_feat), 5)/torch.prod(torch.arange(1, 6).float())
#         #     conf_feat_taylor += self.conf_head[i](conf_feat)
        
#         return loc_feat_taylor, conf_feat_taylor