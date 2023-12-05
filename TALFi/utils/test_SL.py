import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from TALFi.dataset.Classify_dataloader import get_video_info, get_class_index_map
from TALFi.modeling.meta_model2 import SLmodel
from TALFi.config import config
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_thresh = config['testing']['nms_thresh']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
checkpoint = config['testing']['checkpoint_path']
output_path = config['testing']['output_path']
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    
    for epoch in range(1, max_epoch+1):
        print("epoch: ", epoch)
        checkpoint_path = checkpoint +'checkpoint-' + str(epoch) + '.ckpt'
        video_infos = get_video_info(config['dataset']['testing']['csi_info_path'])
        originidx_to_idx, idx_to_class = get_class_index_map()

        npy_data_path = config['dataset']['testing']['csi_data_path']

        net = SLmodel()
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()

        score_func = nn.Softmax(dim=-1)

        result_dict = {}
        for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
            sample_count = video_infos[video_name]['sample_count']
            sample_fps = video_infos[video_name]['sample_fps']

            # 计算滑窗
            if sample_count < clip_length:
                offsetlist = [0]
            else:
                offsetlist = list(range(0, sample_count - clip_length + 1, stride))
                if (sample_count - clip_length) % stride:
                    offsetlist.append(sample_count - clip_length)

            data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
            data = np.transpose(data)
            data = torch.from_numpy(data)

            video_results = []
            for offset in offsetlist:
                clip = data[:, offset: offset + clip_length]
                clip = clip.float()
                clip = clip / 40.0  # 假设这是必要的归一化步骤
                clip = clip.unsqueeze(0).cuda()

                with torch.no_grad():
                    output_dict = net(clip)

                conf = output_dict['conf']
                conf = score_func(conf)

                # 处理每个类别的得分
                for cl in range(1, num_classes):
                    class_score = conf[0, cl]
                    if class_score > conf_thresh:
                        start_time = float(offset) / sample_fps
                        end_time = float(offset + clip_length) / sample_fps
                        video_results.append({
                            'label': idx_to_class[cl],
                            'score': class_score.item(),
                            'segment': [start_time, end_time]
                        })

            # 每个视频的结果
            result_dict[video_name] = video_results

        # 输出结果
        output_dict = {"version": "THUMOS14", "results": result_dict, "external_data": {}}
        json_name = "checkpoint" + str(epoch) + ".json"
        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)