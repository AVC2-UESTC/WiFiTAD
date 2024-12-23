
import torch
import torch.nn as nn
from TAD.model.modules2 import ScaleExp, Transformer_encoder
from TAD.model.prediction_head import detection_head
from TAD.config import config
from TAD.model.modules1 import dee, GaussianSampling
num_classes = config['dataset']['num_classes']
layer_num = 4
priors = 128


class Pyramid_Detection(nn.Module):
    def __init__(self):
        super(Pyramid_Detection, self).__init__()
        self.layer_num = layer_num
        self.pyramids = nn.ModuleList()
        self.small = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        
        for i in range(layer_num):
            self.pyramids.append(dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=priors//(2**i)))
            self.small.append(GaussianSampling(len=priors//(2**i),r=(2048//priors)*(2**i)))
            
        self.detection_head = detection_head()
        self.priors = []
        t = priors
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        
    def forward(self, feat_dict, global_feat):
        pyramid_feats = []
        locs = []
        confs = []
        x = feat_dict
        global_feat = global_feat.detach()
        batch_num = x.size(0)
        for i in range(len(self.pyramids)):
            x = self.pyramids[i](x)
            p = self.small[i](x, global_feat)
            pyramid_feats.append(p)
        
        for i, feat in enumerate(pyramid_feats):
            loc_logits, conf_logits = self.detection_head(feat)
            locs.append(
                self.loc_heads[i](loc_logits)
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )
            confs.append(
                conf_logits.view(batch_num, num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, priors