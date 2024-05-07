# import torch
# import torch.nn as nn
# from TALFi.modeling.ku2 import ScaleExp, Transformer_encoder
# from TALFi.modeling.detection import detection_head
# from TALFi.config import config
# num_classes = config['dataset']['num_classes']
# layer_num = 8
# priors = 128
# import torch.nn.functional as F

# class LayerNorm(nn.Module):
#     """
#     LayerNorm that supports inputs of size B, C, T
#     """

#     def __init__(
#             self,
#             num_channels,
#             eps=1e-5,
#             affine=True,
#             device=None,
#             dtype=None,
#     ):
#         super().__init__()
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.num_channels = num_channels
#         self.eps = eps
#         self.affine = affine

#         if self.affine:
#             self.weight = nn.Parameter(
#                 torch.ones([1, num_channels, 1], **factory_kwargs))
#             self.bias = nn.Parameter(
#                 torch.zeros([1, num_channels, 1], **factory_kwargs))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#     def forward(self, x):
#         assert x.dim() == 3
#         assert x.shape[1] == self.num_channels

#         # normalization along C channels
#         mu = torch.mean(x, dim=1, keepdim=True)
#         res_x = x - mu
#         sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
#         out = res_x / torch.sqrt(sigma + self.eps)

#         # apply weight and bias
#         if self.affine:
#             out *= self.weight
#             out += self.bias

#         return out

# class SGPBlock(nn.Module):
#     """
#     A simple conv block similar to the basic block used in ResNet
#     """

#     def __init__(
#             self,
#             n_embd=512,  # dimension of the input features
#             kernel_size=3,  # conv kernel size
#             n_ds_stride=2,  # downsampling stride for the current layer
#             k=3,  # k
#             group=1,  # group for cnn
#             n_out=None,  # output dimension, if None, set to input dim
#             n_hidden=None,  # hidden dim for mlp
#             path_pdrop=0.0,  # drop path rate
#             act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
#             downsample_type='avg',
#             init_conv_vars=1  # init gaussian variance for the weight
#     ):
#         super().__init__()
#         # must use odd sized kernel
#         # assert (kernel_size % 2 == 1) and (kernel_size > 1)
#         # padding = kernel_size // 2

#         self.kernel_size = kernel_size
#         self.stride = n_ds_stride

#         if n_out is None:
#             n_out = n_embd

#         self.ln = LayerNorm(n_embd)

#         self.gn = nn.GroupNorm(16, n_embd)

#         assert kernel_size % 2 == 1
#         # add 1 to avoid have the same size as the instant-level branch
#         up_size = round((kernel_size + 1) * k)
#         up_size = up_size + 1 if up_size % 2 == 0 else up_size

#         self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
#         self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
#         self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
#         self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
#         self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

#         # input
#         if n_ds_stride > 1:
#             if downsample_type == 'max':
#                 kernel_size, stride, padding = \
#                     n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
#                 self.downsample = nn.MaxPool1d(
#                     kernel_size, stride=stride, padding=padding)
#                 self.stride = stride
#             elif downsample_type == 'avg':
#                 self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
#                                                 nn.Conv1d(n_embd, n_embd, 1, 1, 0))
#                 self.stride = n_ds_stride
#             else:
#                 raise NotImplementedError("downsample type error")
#         else:
#             self.downsample = nn.Identity()
#             self.stride = 1

#         # two layer mlp
#         if n_hidden is None:
#             n_hidden = 4 * n_embd  # default
#         if n_out is None:
#             n_out = n_embd

#         self.mlp = nn.Sequential(
#             nn.Conv1d(n_embd, n_hidden, 1, groups=group),
#             act_layer(),
#             nn.Conv1d(n_hidden, n_out, 1, groups=group),
#         )

#         # drop path
#         if path_pdrop > 0.0:
#             self.drop_path_out = nn.Dropout(path_pdrop)
#             self.drop_path_mlp = nn.Dropout(path_pdrop)
#         else:
#             self.drop_path_out = nn.Identity()
#             self.drop_path_mlp = nn.Identity()

#         self.act = act_layer()
#         self.reset_params(init_conv_vars=init_conv_vars)

#     def reset_params(self, init_conv_vars=0):
#         torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
#         torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
#         torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
#         torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
#         torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
#         torch.nn.init.constant_(self.psi.bias, 0)
#         torch.nn.init.constant_(self.fc.bias, 0)
#         torch.nn.init.constant_(self.convw.bias, 0)
#         torch.nn.init.constant_(self.convkw.bias, 0)
#         torch.nn.init.constant_(self.global_fc.bias, 0)

#     def forward(self, x):
#         # X shape: B, C, T
#         B, C, T = x.shape
#         x = self.downsample(x)
#         out = self.ln(x)
#         psi = self.psi(out)
#         fc = self.fc(out)
#         convw = self.convw(out)
#         convkw = self.convkw(out)
#         phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
#         out = fc * phi + (convw + convkw) * psi + out

#         out = x + self.drop_path_out(out)
#         # FFN
#         out = out

#         return out

# class Pyramid_Detection(nn.Module):
#     def __init__(self):
#         super(Pyramid_Detection, self).__init__()
#         self.layer_num = layer_num
#         self.pyramids = nn.ModuleList()
#         self.loc_heads = nn.ModuleList()
        
#         for i in range(layer_num):
#             self.pyramids.append(SGPBlock())
            
#         self.detection_head = detection_head()
#         self.priors = []
#         t = priors
#         for i in range(layer_num):
#             self.loc_heads.append(ScaleExp())
#             self.priors.append(
#                 torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
#             )
#             t = t // 2
        
#     def forward(self, feat_dict):
#         pyramid_feats = []
#         locs = []
#         confs = []
#         x = feat_dict
#         # origin = feat_dict
#         batch_num = x.size(0)
        
#         for i in range(len(self.pyramids)):
#             x = self.pyramids[i](x)
#             pyramid_feats.append(x)
        
#         for i, feat in enumerate(pyramid_feats):
#             loc_logits, conf_logits = self.detection_head(feat)
#             locs.append(
#                 self.loc_heads[i](loc_logits)
#                     .view(batch_num, 2, -1)
#                     .permute(0, 2, 1).contiguous()
#             )
#             confs.append(
#                 conf_logits.view(batch_num, num_classes, -1)
#                     .permute(0, 2, 1).contiguous()
#             )

#         loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
#         conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
#         priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
#         return loc, conf, priors

import torch
import torch.nn as nn
from TALFi.modeling.ku2 import ScaleExp, Transformer_encoder
from TALFi.modeling.detection import detection_head
from TALFi.config import config
num_classes = config['dataset']['num_classes']
layer_num = 8
priors = 128

class Pyramid_Detection(nn.Module):
    def __init__(self):
        super(Pyramid_Detection, self).__init__()
        self.layer_num = layer_num
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        
        for i in range(layer_num):
            self.pyramids.append(Transformer_encoder(512, 512, 4,2048,1, i))
            
        self.detection_head = detection_head()
        self.priors = []
        t = priors
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        
    def forward(self, feat_dict):
        pyramid_feats = []
        locs = []
        confs = []
        x = feat_dict
        origin = feat_dict
        batch_num = x.size(0)
        
        for i in range(len(self.pyramids)):
            x, origin = self.pyramids[i](x, origin)
            pyramid_feats.append(x)
        
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
