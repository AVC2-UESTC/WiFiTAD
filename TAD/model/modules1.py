
import torch
import torch.nn as nn
from TAD.model.modules2 import Unit1D
from TAD.model.atten import FullAttention, AttentionLayer, FullAttention_new, Cross_Attention
from TAD.model.encoder import Encoder2, EncoderLayer2, Encoder3, EncoderLayer3
    
class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x
        
class PoolConv(nn.Module):
    def __init__(self, in_channels):
        super(PoolConv, self).__init__()
        self.dwconv1 = nn.Sequential(
            Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None),
            nn.ReLU(inplace=True))
        self.max = nn.Sequential(nn.MaxPool1d(3,2,1), nn.Sigmoid())
        self.conv = Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=2,
                        use_bias=True,
                        activation_fn=None)
        self.conv2 = Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None)
        self.norm = nn.GroupNorm(32, 512)
        self.lu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.dwconv1(x)
        y = self.norm(self.max(y)*self.conv(y))
        y = self.conv2(y)

        return  y
    
class DSB(nn.Module):
    def __init__(self, in_channels):
        super(DSB, self).__init__()
        self.dwconv1 = nn.Sequential(Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=2,
                        use_bias=True,
                        activation_fn=None),
                        nn.GroupNorm(32, 512),
                        nn.ReLU(inplace=True)
                        )

    def forward(self, x):
        x = self.dwconv1(x)
        return  x

class Cat_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cat_Fusion, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, 1, bias=True),
                        Unit1D(in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=False,
                        activation_fn=None),
                        nn.PReLU())

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-2)
        x = self.conv(x)
        return  x

class dee(nn.Module):
    def __init__(self, in_channels, out_channels, length, kernel_size=3, stride=2):
        super(dee, self).__init__()
        self.Downscale = DSB(512)

        self.pconv = PoolConv(512)
        self.self_attention = joint_attention(enc_in=in_channels, d_model=in_channels, n_heads=16,length=length, d_ff=in_channels*4, e_layers=1, factor=3, dropout=0.01, output_attention=False, attn='full')
        
        self.c1 = Cat_Fusion(1024, 1024)
        self.c2 = Cat_Fusion(1024, 1024)
        self.c3 = Cat_Fusion(2048, 512)
        self.contra_norm = ContraNorm(dim=length, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False)

    def forward(self, time):
        high = self.pconv(time)
        time2 = self.Downscale(time)
        low = self.self_attention(time2, time2, time2)
        high2 = self.c1(low, high)
        low2 = self.c2(high, low)
        out = self.c3(high2, low2)
        out = self.contra_norm(out)
        return out
    
class joint_attention(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, length, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(joint_attention, self).__init__()
        if attn == "full":
            self.cross_atten = Encoder2(
                [
                    EncoderLayer2(
                        AttentionLayer(FullAttention_new(len=length, n_heads=n_heads, attention_dropout=dropout, output_attention=output_attention), 
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                norm_layer=None
            )
        else:
            self.cross_atten = Encoder2(
                [
                    EncoderLayer2(
                        AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                norm_layer=None
            )
    
    def forward(self, q, k, v):
        out, _ = self.cross_atten(q, k, v, attn_mask=None)
        out = out.permute(0, 2, 1)
        return out
    
class crsa(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, length, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(crsa, self).__init__()
        self.cross_atten = Encoder3(
            [
                EncoderLayer3(
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=None
        )
    
    def forward(self, feat1, feat2):
        out, _ = self.cross_atten(feat1, feat2, attn_mask=None)
        out = out.permute(0, 2, 1)
        return out
    
class GaussianSampling(nn.Module):
    def __init__(self, len,r):
        super(GaussianSampling, self).__init__()
        self.r = r
        self.len = len
        self.conv = nn.Conv1d(512,512,1,1)
        self.beta = nn.Parameter(torch.ones(1))
        self.crs = crsa(length=1,enc_in=512, d_model=512, n_heads=16, d_ff=512*4, e_layers=1, factor=3, dropout=0.01, output_attention=False, attn='full2')


    def forward(self, x, global_feat):
        return self.sliding_window_gaussian(x, global_feat, self.r)
    
    def sliding_window_gaussian(self, feat, global_feat, r):
        batch_size, channels, length = global_feat.shape
        step = 2 * r
        window_size = 2 * r
        
        num_windows = (length - window_size) // step + 1
        result = torch.zeros(batch_size, channels, num_windows, dtype=torch.float32).to(feat.device)
        
        for i in range(0, length - window_size + 1, step):
            window = global_feat[:, :, i:i + window_size]  # Extract window directly
            
            # Calculate maximum and minimum values in the window
            window_max = window.max(dim=-1, keepdim=True)[0]
            window_min = window.min(dim=-1, keepdim=True)[0]
            
            # Calculate Euclidean distance between max and min values
            euclidean_dist = torch.norm(window_max - window_min, p=2, dim=-1)
            
            # Determine corresponding index in result tensor
            idx = i // step
            
            # Store the Euclidean distance values in result tensor
            result[:, :, idx] = euclidean_dist
        
        # Apply your subsequent operations
        result = torch.relu(self.conv(result))
        new_feat = self.crs(feat, result)
        # new_feat = self.crs1(feat, result, result) + self.crs2(result, feat, feat)
        # new_feat = self.crs3(new_feat.permute(0,2,1)).permute(0,2,1)
        # new_feat = new_feat
        # print(new_feat.shape)
        return new_feat
