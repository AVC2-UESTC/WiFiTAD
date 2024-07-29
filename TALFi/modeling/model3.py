import torch.nn as nn
import numpy as np
from TALFi.modeling.backbone import Backbone, Embedding
from TALFi.modeling.pyramid3 import Pyramid_Detection
from TALFi.modeling.module3 import dee
conv_channels = 512

class NETloader(nn.Module):
    def __init__(self):
        super(NETloader, self).__init__()
        self.adaptive_pool = nn.ModuleList([
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=2048),
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=1024),
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=512),
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=256),
                        #   dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=128),
                          ])
        
    def forward(self, time):
        for block in self.adaptive_pool:
            time = block(time)
        return time

class talfi(nn.Module):
    def __init__(self, in_channels=30):
        super(talfi, self).__init__()
        self.embedding = Embedding(30)
        self.backbone = NETloader()
        self.pyramid_detection = Pyramid_Detection()
        self.reset_params()
        
    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.embedding(x)
        p = x
        x = self.backbone(x)
        loc, conf, priors = self.pyramid_detection(x, p)
        return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }


