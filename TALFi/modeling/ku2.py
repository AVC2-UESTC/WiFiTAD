import torch
import torch.nn as nn
import torch.nn.functional as F
from TALFi.modeling.embedding import DatawithoutPOSEmbedding
from TALFi.modeling.atten import FullAttention, AttentionLayer
from TALFi.modeling.encoder import Encoder, EncoderLayer, Encoder2, EncoderLayer2, EncoderLayer_Hou

class ScaleExp(nn.Module):
    '''
    Different layers regression to different size range
    Learn a trainable scalar to automatically adjust the base of exp(si * x)
    '''
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)

class Unit1D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=1,
                 stride=1,
                 padding='same',
                 activation_fn=F.relu,
                 use_bias=True):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                output_channels,
                                kernel_shape,
                                stride,
                                padding=0,
                                bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - (t % self._stride), 0)

    def forward(self, x):
        if self._padding == 'same':
            batch, channel, t = x.size()
            pad_t = self.compute_pad(t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class Projection(nn.Module):
    '''
    Increase Channel
    '''
    def __init__(self, in_channels):
        super(Projection, self).__init__()
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.project(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.squeeze(x)
        se = self.excitation(se)
        return x * se.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, downsample=True, SE=True):
        super(ResidualSEBlock, self).__init__()
        self.downsample = downsample
        self.SE = SE
        self.Res_Block = nn.Sequential(
                Unit1D(in_channels, out_channels, 3, activation_fn=None),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
                Unit1D(out_channels, out_channels, 3, activation_fn=None),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
                Unit1D(out_channels, out_channels, 3, activation_fn=None),
                nn.GroupNorm(32, out_channels)
        )
        
        self.SE_Block = nn.Sequential(nn.Conv1d(out_channels,out_channels, 1, 1, 0),  SEBlock(512), nn.GroupNorm(32, out_channels),nn.ReLU(inplace=True))
        
        self.Downscale_Block = nn.Sequential(
                            Unit1D(
                                in_channels=out_channels,
                                output_channels=out_channels,
                                kernel_shape=kernel_size,
                                stride=stride,
                                use_bias=True,
                                activation_fn=None
                            ),
                            nn.GroupNorm(32, 512),
                            nn.ReLU(inplace=True))

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.Res_Block(x) + x)
        if self.SE== True:
            x = self.SE_Block(x)
        if self.downsample==True:
            x = self.Downscale_Block(x)
        return x

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, downsample=True):
        super(ResidualBlock2, self).__init__()
        channels = out_channels//2
        self.downsample = downsample
        self.res_branch = nn.Sequential(nn.Conv1d(channels, channels, 3, 1, 1), 
                                 nn.GroupNorm(32, channels),
                                 nn.GELU(), 
                                 nn.Conv1d(channels, channels, 1, 1, 0),
                                 nn.GroupNorm(32, channels),
                                 nn.GELU())

        self.max_branch = nn.Sequential(nn.MaxPool1d(kernel_size=5, stride=1, padding=2), 
                                        nn.Conv1d(channels, channels, 3, 1, 1),
                                        nn.GroupNorm(32, channels),
                                        nn.GELU())
        
        self.conv_down = nn.Sequential(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding), 
                                       nn.GroupNorm(32, out_channels), 
                                       nn.ReLU(inplace=True))

        self.fusion = nn.Sequential(nn.Conv1d(out_channels,out_channels, 1, 1, 0),  SEBlock(512),nn.GroupNorm(32, out_channels),nn.ReLU(inplace=True))
        self.glu = nn.GELU()
        
    def forward(self, x):
        x1 = x[:, :256, :]
        x2 = x[:, 256:, :]
        
        x1 = self.res_branch(x1)
        x2 = self.max_branch(x2)
        out = torch.cat((x1, x2), dim=1)
        out = self.glu(self.fusion(out)+x)
        if self.downsample==True:
            out = self.conv_down(out)
        return out

class CSABlock(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, length, reduction=4, factor=3, dropout=0.01, output_attention=False, attn='full', activation='gelu', distil=False):
        super(CSABlock, self).__init__()
        self.reduction= reduction
        self.embedding_SA = DatawithoutPOSEmbedding(enc_in, d_model, 1, dropout)
        self.embedding_channel = DatawithoutPOSEmbedding(length, length, 1, dropout)
        Attn = FullAttention
        self.activation = F.relu if activation == "relu" else F.gelu
        self.conv_downsample = nn.Sequential(Unit1D(
                    in_channels=512,
                    output_channels=512,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),nn.GroupNorm(32, 512),nn.ReLU(inplace=True))
        self.self_attention = Encoder(
            [
                EncoderLayer_Hou(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=None
        )
        self.channel_attention = Encoder(
            [
                EncoderLayer_Hou(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                length, n_heads, mix=False),
                    length,
                    length*4,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=None
        )
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(enc_in, enc_in // self.reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(enc_in // self.reduction, enc_in, kernel_size=1),
            nn.Sigmoid()
        )
        self.dwconv = nn.Conv1d(512, 512, 3, 1, 1, groups=512)
        self.gn = nn.GroupNorm(32, 512)
    def forward(self, x_enc):
        x_enc = self.conv_downsample(x_enc)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.embedding_SA(x_enc)
        x_self, _ = self.self_attention(x_enc, attn_mask=None)
        x_self = x_self.permute(0, 2, 1)
        x_self = self.dwconv(x_self)
        
        x_channel = x_self
        x_channel, _ = self.channel_attention(x_channel, attn_mask=None)
        x_se = self.squeeze(x_channel)
        x_se = self.excitation(x_se)
        return self.gn(x_self*x_se)
    
class Transformer_encoder2(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(Transformer_encoder2, self).__init__()
        self.enc_embedding_informer = DatawithoutPOSEmbedding(enc_in, d_model, 1, dropout)
        self.enc_embedding_autoformer = DatawithoutPOSEmbedding(enc_in, d_model, 1, dropout)
        Attn = FullAttention
        self.infomer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.conv_downsample = nn.Sequential(Unit1D(
                    in_channels=512,
                    output_channels=512,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),nn.GroupNorm(32, 512),nn.ReLU(inplace=True))
    def forward(self, x_enc):
        x_enc = self.conv_downsample(x_enc)
        x_enc = x_enc.permute(0, 2, 1)
        info = self.enc_embedding_informer(x_enc)
        info_out, _ = self.infomer_encoder(info, attn_mask=None)
        enc_out = info_out.permute(0, 2, 1)
        return enc_out

class joint_attention(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(joint_attention, self).__init__()
        self.enc_embedding_informer = DatawithoutPOSEmbedding(enc_in, d_model, 1, dropout)
        self.enc_embedding_informer2 = DatawithoutPOSEmbedding(enc_in, d_model, 1, dropout)
        Attn = FullAttention
        self.infomer_encoder = Encoder2(
            [
                EncoderLayer2(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
    def forward(self, x_enc, k):
        x_enc = x_enc.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        info = self.enc_embedding_informer(x_enc)
        info_k = self.enc_embedding_informer2(k)
        info_out, _ = self.infomer_encoder(info,info_k, attn_mask=None)
        enc_out = info_out.permute(0, 2, 1)
        return enc_out

class Pyramid_layer(nn.Module):
    def __init__(self, downsample=True):
        super(Pyramid_layer, self).__init__()
        self.downsample = downsample
        
        self.deconv = nn.Sequential(
            Unit1D(512, 512, 3, activation_fn=None),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            Unit1D(512, 512, 3, activation_fn=None),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            Unit1D(512, 512, 3, activation_fn=None),
            nn.GroupNorm(32, 512)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_downsample = nn.Sequential(Unit1D(
                    in_channels=512,
                    output_channels=512,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),nn.GroupNorm(32, 512),nn.ReLU(inplace=True))
        
    def forward(self, x_enc):
        enc_out = self.relu(self.deconv(x_enc)+x_enc)
        if self.downsample == True:
            enc_out = self.conv_downsample(enc_out)
        return enc_out
