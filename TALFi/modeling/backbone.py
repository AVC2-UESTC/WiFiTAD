import torch
import torch.nn as nn
from TALFi.modeling.ku2 import Projection, ResidualSEBlock, joint_attention

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1,((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1,(self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size=25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        
        fft_output = torch.fft.fft(x, dim=2)
        amplitude = torch.abs(fft_output)
        phase = torch.angle(fft_output)
        normalized_amplitude = (amplitude - amplitude.mean()) / amplitude.std()
        
        data = torch.cat((x, res), dim=-2)
        fft_data = torch.cat([normalized_amplitude, phase], dim=1)
        return data, fft_data

class Downsample_Net(nn.Module):
    def __init__(self, in_channels):
        super(Downsample_Net, self).__init__()
        self.project_layer = Projection(in_channels)
        self.adaptive_pool = nn.ModuleList([
                          ResidualSEBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
                          ResidualSEBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
                          ResidualSEBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
                          ResidualSEBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)])
        
    def forward(self, x):
        x = self.project_layer(x)
        for block in self.adaptive_pool:
            x = block(x)
        return x
    
class Downsample_Net2(nn.Module):
    def __init__(self, in_channels):
        super(Downsample_Net2, self).__init__()
        self.project_layer = Projection(in_channels)
        self.adaptive_pool = nn.ModuleList([
                          ResidualSEBlock(in_channels=512, out_channels=512, kernel_size=7, stride=4, padding=1),
                          ResidualSEBlock(in_channels=512, out_channels=512, kernel_size=7, stride=4, padding=1),])
        
    def forward(self, x):
        x = self.project_layer(x)
        for block in self.adaptive_pool:
            x = block(x)
        return x

class Backbone(nn.Module):
    def __init__(self, in_channels):
        super(Backbone, self).__init__()
        self.temporal_stream = Downsample_Net(in_channels)
        self.frequency_stream = Downsample_Net2(in_channels)
        self.decomp = series_decomp(25)
        self.JA = joint_attention(enc_in=512, d_model=512, n_heads=8, d_ff=2048, e_layers=6, factor=3, dropout=0.1, output_attention=False)
        
    def forward(self, x):
        temporal_x, fft_x = self.decomp(x)
        temporal_feat = self.temporal_stream(temporal_x)
        fft_feat = self.frequency_stream(fft_x)
        # joint_feat = self.JA(temporal_feat, fft_feat)
        joint_feat = self.JA(fft_feat, temporal_feat)
        return joint_feat