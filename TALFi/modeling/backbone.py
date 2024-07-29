import torch
import torch.nn as nn
from TALFi.modeling.ku2 import TFBlock, SeeB
import matplotlib.pyplot as plt

class Embedding(nn.Module):
    def __init__(self, in_channels):
        super(Embedding, self).__init__()
        self.embedding = nn.Sequential(
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
        
    def forward(self, time):
        time = self.embedding(time)
        return time

# class Backbone(nn.Module):
#     def __init__(self):
#         super(Backbone, self).__init__()
#         self.adaptive_pool = nn.ModuleList([
#                           TFBlock(in_channels=512, out_channels=512,kernel_size=3, stride=2),
#                           TFBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
#                           TFBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
#                           TFBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
#                           ])
        
#     def forward(self, time):
#         for block in self.adaptive_pool:
#             time = block(time)
#         return time
import torch.nn.functional as F
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
    
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.adaptive_pool = nn.ModuleList([
                          SeeB(in_channels=512, out_channels=512,kernel_size=3, stride=2),
                          SeeB(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                          SeeB(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                          SeeB(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                          ])
        self.se = SEBlock(channels=512)
        
    def forward(self, time):
        avg = time
        max = time
        channel = time

        channel = self.se(channel)
        for block in self.adaptive_pool:
            avg, max = block(avg, max)
        output = (avg + max) * channel
        return output


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
