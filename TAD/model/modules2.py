import torch
import torch.nn as nn
import torch.nn.functional as F
from TAD.model.embedding import DatawithoutPOSEmbedding
from TAD.model.atten import FullAttention, AttentionLayer, FullAttention2, AttentionLayer2, CrossAttention, CrossAttentionLayer
from TAD.model.encoder import Encoder, EncoderLayer, Encoder2, EncoderLayer2, CrossEncoderLayer


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

class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=2, stride=2):
        super(ConvBlock, self).__init__()
        self.dw = nn.Conv1d(channels, channels, 3,1, 1)
        self.maxdw = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(channels, channels, 3,1, 1, groups=channels))
        self.lu = nn.SiLU(inplace=True)

    def forward(self, x):
        out = x + self.dw(x) + self.maxdw(x)
        out = self.lu(out)
        return out

class ffn(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=3, stride=2):
        super(ffn, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.groupnorm = nn.GroupNorm(32,512)
        self.ffn1 = Unit1D(
                                in_channels=in_channels,
                                output_channels=2048,
                                kernel_shape=kernel_size,
                                stride=1,
                                use_bias=True,
                                activation_fn=None
                            )
        self.ffn2 = Unit1D(
                                in_channels=2048,
                                output_channels=out_channels,
                                kernel_shape=kernel_size,
                                stride=1,
                                use_bias=True,
                                activation_fn=None
                            )


    def forward(self, x):
        x1 = self.groupnorm(x)
        x1 = (self.ffn2(self.relu(self.ffn1(x1)))) + x
        return x1


class Transformer_encoder(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, layer, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(Transformer_encoder, self).__init__()
        self.enc_embedding_informer = DatawithoutPOSEmbedding(enc_in, d_model, 1, dropout)
        self.enc_embedding_informer2 = DatawithoutPOSEmbedding(enc_in, d_model, 1, dropout)

        self.down_kernal = 2**layer
        self.down_stirde = self.down_kernal
        self.down_padding = self.down_kernal/2
        self.up_kernal = 2**layer
        self.up_stride = self.up_kernal
        self.up_padding = self.up_kernal/2
        self.add_gate = nn.Sequential(nn.Conv1d(512, 512, self.down_kernal, self.down_stirde, 0), nn.Sigmoid())
        self.renew_gate = nn.Sequential(nn.ConvTranspose1d(512, 512, self.up_kernal, self.up_stride, 0), nn.Sigmoid())
        self.refine = nn.Sequential(nn.Conv1d(512, 512, 3, 1, 1), nn.ReLU(inplace=True), nn.GroupNorm(32, 512))

        Attn = FullAttention2
        self.infomer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer2(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=None
        )
        self.conv_downsample = nn.Sequential(Unit1D(
                    in_channels=512,
                    output_channels=512,
                    kernel_shape=3,
                    stride=1,
                    use_bias=True,
                    activation_fn=None
                ),nn.GroupNorm(32, 512),nn.MaxPool1d(2,2))
    def forward(self, x_enc, origin):

        origin_add = self.add_gate(origin)
        # x_enc = x_enc*origin_add

        x_enc = x_enc.permute(0, 2, 1)
        origin_add = origin_add.permute(0, 2, 1)
        info = self.enc_embedding_informer(x_enc)
        info2 = self.enc_embedding_informer2(origin_add)
        info_out, _ = self.infomer_encoder(info, info2, attn_mask=None)
        enc_out = info_out.permute(0, 2, 1)

        origin = self.refine(origin*self.renew_gate(enc_out))

        enc_out = self.conv_downsample(enc_out)
        return enc_out, origin
    
class joint_attention(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(joint_attention, self).__init__()
        if attn == "full":
            # en = EncoderLayer2
            an = FullAttention
            attlay = AttentionLayer
        else:
            # en = CrossEncoderLayer
            an = CrossAttention
            attlay = CrossAttentionLayer

        self.cross_atten = Encoder2(
            [
                EncoderLayer2(
                    attlay(an(False, factor, attention_dropout=dropout, output_attention=output_attention), 
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
