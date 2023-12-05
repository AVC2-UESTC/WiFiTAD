import torch.nn as nn
import numpy as np
from TALFi.modeling.backbone import Backbone
from TALFi.modeling.multi_temporal_pyramid import Pyramid_Detection
conv_channels = 512

class talfi(nn.Module):
    def __init__(self, in_channels=60):
        super(talfi, self).__init__()
        self.pyramid_detection = Pyramid_Detection()
        self.backbone = Backbone(in_channels)
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
        joint_feat = self.backbone(x)
        loc, conf, priors = self.pyramid_detection(joint_feat)
        return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }
