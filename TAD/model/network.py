import torch.nn as nn
import numpy as np
from TAD.model.embedding import Embedding
from TAD.model.feature_extractor import Pyramid_Detection
from TAD.model.modules1 import dee
conv_channels = 512

class NETloader(nn.Module):
    def __init__(self):
        super(NETloader, self).__init__()
        self.adaptive_pool = nn.ModuleList([
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=2048),
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=1024),
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=512),
                          dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=256),
                        #   dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=128),
                        #   dee(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=64),
                          ])
        
    def forward(self, time):
        for block in self.adaptive_pool:
            time = block(time)
        return time

class talfi(nn.Module):
    def __init__(self, in_channels=30):
        super(talfi, self).__init__()
        self.embedding = Embedding(30)
        self.backbone = NETloader()
        self.pyramid_detection = Pyramid_Detection()
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
        x = self.embedding(x)
        p = x
        x = self.backbone(x)
        loc, conf, priors = self.pyramid_detection(x, p)
        return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }


# # # from ptflops import get_model_complexity_info
# from fvcore.nn import FlopCountAnalysis
# import torch.nn as nn
# import torch
# import time
# def test_inference(model, sampling_rate=100, length=4096, repeats=200, warmup_times=5):
#     run_times = []
#     video_time = length / sampling_rate
#     num_frame = video_time * sampling_rate
#     input_tensor = torch.rand(1, 30, length)
    
#     # 使用FlopCountAnalysis分析FLOPs
#     flop_counter = FlopCountAnalysis(model, input_tensor)
#     flops = flop_counter.total() / 1e9
#     print(f"FLOPs (G): {flops:.2f}G")

#     # 输出参数统计，转换为Million (10^6)
#     param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
#     print(f"Params (M): {param_count:.2f}M")

#     x = input_tensor
#     for i in range(repeats + warmup_times):
#         torch.cuda.synchronize()
#         start = time.time()
#         with torch.no_grad():
#             y = model(x)
#         torch.cuda.synchronize()
#         run_times.append(time.time() - start)

#     infer_time = np.mean(run_times[warmup_times:])
#     infer_fps = num_frame * (1.0 / infer_time)
#     print("inference time (ms):", infer_time * 1000)
#     print("infer_fps:", int(infer_fps))

# # 初始化模型和输入数据
# model = talfi()  # 假设模型输入为 (1, 30, 4096)

# # 调用函数进行测试
# test_inference(model, repeats=500, warmup_times=5, sampling_rate=100, length=4096)