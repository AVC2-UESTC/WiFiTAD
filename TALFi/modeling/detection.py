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

    def forward(self, x):
        duration = self.duration(x)
        center = self.center(x)
        w_duration = duration.repeat(1,2,1)
        w_duration[:, 0, :] = w_duration[:, 0, :] - center/2
        w_duration[:, 1, :] = w_duration[:, 1, :] + center/2
        return w_duration
    
    
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


class detection_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_tower = Tower(512, 3)
        self.conf_tower = Tower(512, 3)
        
        self.loc_head = loc_head_af()
        self.conf_head = conf_head_af()
        
    def forward(self, x):
        loc_feat = self.loc_tower(x)
        conf_feat = self.conf_tower(x)
        
        loc_feat = self.loc_head(loc_feat)
        conf_feat = self.conf_head(conf_feat)
        
        return loc_feat, conf_feat


class loc_head_ceshipiancha(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.zuoyou = nn.Sequential(
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
                output_channels=2,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None)
            )
        
    def forward(self, x):
        duration = self.zuoyou(x)
        return duration