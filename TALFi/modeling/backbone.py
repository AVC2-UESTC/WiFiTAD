import torch
import torch.nn as nn
from TALFi.modeling.ku2 import Projection, TFBlock
import matplotlib.pyplot as plt

class Downsample_Net(nn.Module):
    def __init__(self, in_channels):
        super(Downsample_Net, self).__init__()
        self.project_layer1 = Projection(in_channels)
        # self.project_layer2 = Projection(in_channels)
        # self.project_layer3 = Projection(in_channels)
        self.adaptive_pool = nn.ModuleList([
                          TFBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                          TFBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                          TFBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                          TFBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                          ])
        
    def forward(self, time):
        time = self.project_layer1(time)
        # fre = self.project_layer2(fre)
        # ang = self.project_layer3(ang)
        for block in self.adaptive_pool:
            time = block(time)
        return time

class Backbone(nn.Module):
    def __init__(self, in_channels):
        super(Backbone, self).__init__()
        self.TFB_Stream = Downsample_Net(in_channels=30)

    def forward(self, x):
        # plot_and_save_csi(x, 'wifi_csi.png')
        # frequency_domain = torch.fft.fft(x, dim=2)
        # amplitude = torch.abs(frequency_domain)
        # angle = torch.angle(frequency_domain)
        # 将频率大于200的部分置零
        # frequency_domain[..., 1000:] = 0
        # amplitude[..., :1] = 0

        feat = self.TFB_Stream(x)

        return feat



def plot_and_save_csi(data, save_path):
    # 获取batch中的第一个样本，假设我们仅关注一个样本
    real_sample = data[0]  # 提取实部

    # 画矩阵图
    plt.imshow(real_sample.cpu().numpy(), cmap='viridis', aspect='20')

    # 取消轴上的数字
    plt.xticks([])
    plt.yticks([])

    # 设置滚动条
    # plt.colorbar(label='amplitude')
    
    # 保存到文件
    plt.savefig(save_path)
    plt.close()  # 关闭图像，释放资源
