import torch
import torch.nn as nn
import torch.nn.functional as F
from TALFi.ablation_study.resnet1d import ResNet17, ResNet33, ResNet49, ResNet100, ResNet151
from TALFi.modeling.ku2 import Unit1D
from TALFi.modeling.atten import FullAttention, AttentionLayer, FullAttention2, AttentionLayer2, CrossAttention, CrossAttentionLayer
from TALFi.modeling.encoder import Encoder, EncoderLayer, Encoder2, EncoderLayer2, CrossEncoderLayer

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, DS=True):
        super(ConvNet, self).__init__()
        self.ds = DS
        if self.ds == True:
            self.Downscale_Blockt = nn.Sequential(
                                Unit1D(
                                    in_channels=in_channels,
                                    output_channels=out_channels,
                                    kernel_shape=kernel_size,
                                    stride=stride,
                                    use_bias=True,
                                    activation_fn=None
                                ),
                                nn.GroupNorm(out_channels//16, out_channels),
                                nn.ReLU(inplace=True))

        self.conv = ResNet17(in_samples=4096, in_channels=in_channels, end_channels=out_channels, half_start=False)
    def forward(self, x):
        if self.ds == True:
            x = self.Downscale_Blockt(x)
        out = self.conv(x)
        return out
    

class LSTMNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, DS=True):
        super(LSTMNet, self).__init__()
        self.ds = DS
        if self.ds == True:
            self.Downscale_Blockt = nn.Sequential(
                                Unit1D(
                                    in_channels=in_channels,
                                    output_channels=out_channels,
                                    kernel_shape=kernel_size,
                                    stride=stride,
                                    use_bias=True,
                                    activation_fn=None
                                ),
                                nn.GroupNorm(out_channels//16, out_channels),
                                nn.ReLU(inplace=True))
        self.out_channels = out_channels

        # (batch, seq, feature)
        self.lstm = nn.GRU(input_size=self.out_channels, 
                            hidden_size=self.out_channels, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=False)
    def forward(self, x):
        if self.ds ==True:
            x = self.Downscale_Blockt(x)

        batch, _, _ = x.size()
        x = x.view(batch, -1, 512)
        a, (out) = self.lstm(x)
        a = a.permute(0, 2, 1)
        return a

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
    
    def forward(self, x):
        batch, channels, length = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(batch, channels, 1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return  y


class dee(nn.Module):
    def __init__(self, in_channels, out_channels, len, kernel_size=3, stride=2):
        super(dee, self).__init__()
        self.Downscale_Blockt = nn.Sequential(
                                    Unit1D(
                                        in_channels=in_channels,
                                        output_channels=in_channels,
                                        kernel_shape=kernel_size,
                                        stride=stride,
                                        use_bias=True,
                                        activation_fn=None
                                    ),
                                    nn.GroupNorm(in_channels//16, in_channels),
                                    nn.ReLU(inplace=True))

        self.self_attention = joint_attention(enc_in=out_channels, d_model=out_channels, n_heads=16, d_ff=out_channels*4, e_layers=1, factor=3, dropout=0.01, output_attention=False, attn='full', len=len)
        self.conv1 = nn.Sequential(Unit1D(
                                        in_channels=out_channels,
                                        output_channels=out_channels,
                                        kernel_shape=1,
                                        stride=1,
                                        use_bias=True,
                                        activation_fn=None
                                    ),
                                    nn.GroupNorm(out_channels//16, out_channels),)        
        self.conv2 = nn.Sequential(Unit1D(
                                        in_channels=out_channels,
                                        output_channels=out_channels,
                                        kernel_shape=3,
                                        stride=1,
                                        use_bias=True,
                                        activation_fn=None
                                    ),
                                    nn.GroupNorm(out_channels//16, out_channels),)        
        self.conv3 = nn.Sequential(nn.MaxPool1d(3,1,1),
                                    Unit1D(
                                        in_channels=out_channels,
                                        output_channels=out_channels,
                                        kernel_shape=3,
                                        stride=1,
                                        use_bias=True,
                                        activation_fn=None
                                    ),
                                    nn.GroupNorm(out_channels//16, out_channels),)
        self.se = SEBlock(channels=512)
        self.lu = nn.SiLU(inplace=True)
        self.FFN = nn.Sequential(nn.Conv1d(512, 2048,1,1), nn.ReLU(inplace=True), nn.Conv1d(2048, 512,1,1))
        
    def forward(self, time):
        time = self.Downscale_Blockt(time)
        # se_channel = self.se(time)
        time1 = time[:,:256,:]
        time2 = time[:,256:,:]

        high1 = self.conv1(time1)
        high2 = self.conv2(time1)
        high3 = self.conv3(time1)
        high = self.lu(high1 + high2 + high3)
        low = self.self_attention(time2, time2, time2)
        # channel = self.se(time)
        # high = high * channel
        out = self.lu(torch.cat([high, low], dim=-2) + time)
        out = self.FFN(out)


        return out
    
class joint_attention(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, len, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
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
                    attlay(an(False, factor, d_model, len, attention_dropout=dropout, output_attention=output_attention), 
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


class TransformerEncoderNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, DS=True):
        super(TransformerEncoderNet, self).__init__()
        self.ds = DS
        if self.ds == True:
            self.Downscale_Blockt = nn.Sequential(
                                Unit1D(
                                    in_channels=in_channels,
                                    output_channels=out_channels,
                                    kernel_shape=kernel_size,
                                    stride=stride,
                                    use_bias=True,
                                    activation_fn=None
                                ),
                                nn.GroupNorm(out_channels//16, out_channels),
                                nn.ReLU(inplace=True))
        self.input_channels = in_channels
        self.d_model = in_channels
        
        # 调整输入维度的线性层
        # self.input_linear = nn.Linear(input_channels, d_model)
        
        # Transformer编码器层配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=16, 
            dim_feedforward=self.d_model*4, 
            dropout=0.01,
            batch_first=True  # 注意batch_first参数要设置为True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        if self.ds ==True:
            x = self.Downscale_Blockt(x)
        # 调整维度顺序并变换特征维度
        x = x.permute(0, 2, 1)  # 从(batch_size, channel, length)变为(batch_size, length, channel)
        
        # 通过Transformer Encoder层处理
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        return x


# from TALFi.ablation_study.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
# from TALFi.ablation_study.layers.SelfAttention_Family import FullAttention, AttentionLayer

# def compared_version(ver1, ver2):
#     """
#     :param ver1
#     :param ver2
#     :return: ver1< = >ver2 False/True
#     """
#     list1 = str(ver1).split(".")
#     list2 = str(ver2).split(".")
    
#     for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
#         if int(list1[i]) == int(list2[i]):
#             pass
#         elif int(list1[i]) < int(list2[i]):
#             return -1
#         else:
#             return 1
    
#     if len(list1) == len(list2):
#         return True
#     elif len(list1) < len(list2):
#         return False
#     else:
#         return True
    
# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x
    
# class DataEmbedding_wo_pos(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0):
#         super(DataEmbedding_wo_pos, self).__init__()

#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x):
#         x = x.permute(0,2,1)
#         x = self.value_embedding(x)
#         return self.dropout(x)


# class TransformerModel(nn.Module):
#     """
#     Vanilla Transformer
#     with O(L^2) complexity
#     Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
#     """

#     def __init__(self, e_layers, enc_in=512, factor=5, d_model=512, n_heads=16, d_layers=2, d_ff=2048, 
#                 dropout=0.0, embed='fixed', freq='h', activation='gelu', 
#                 output_attention = False, distil=True, mix=True):
#         super(TransformerModel, self).__init__()
#         # Embedding
#         self.enc_embedding = DataEmbedding_wo_pos(enc_in, 512, embed, freq,
#                                            dropout)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, factor, attention_dropout=dropout,
#                                       output_attention=output_attention), d_model, n_heads),
#                     d_model,
#                     d_ff,
#                     dropout=dropout,
#                     activation=activation
#                 ) for l in range(e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(d_model)
#         )
#     def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
#         # Embedding
#         enc_out = self.enc_embedding(x_enc)

#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         enc_out = enc_out.permute(0,2,1)

#         return enc_out

# class TransformerNet(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, DS=True):
#         super(TransformerNet, self).__init__()
#         self.ds = DS
#         if self.ds == True:
#             self.Downscale_Blockt = nn.Sequential(
#                                 Unit1D(
#                                     in_channels=in_channels,
#                                     output_channels=out_channels,
#                                     kernel_shape=kernel_size,
#                                     stride=stride,
#                                     use_bias=True,
#                                     activation_fn=None
#                                 ),
#                                 nn.GroupNorm(out_channels//16, out_channels),
#                                 nn.ReLU(inplace=True))
#         self.transformer = TransformerModel(e_layers=1)

#     def forward(self, x):
#         if self.ds ==True:
#             x = self.Downscale_Blockt(x)
#         out = self.transformer(x)
#         return out