import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from TAD.dataset.load_csi import get_video_info, get_class_index_map
from TAD.model.tad_model import wifitad
from TAD.evaluation.softnms import softnms_v2
from TAD.config import config
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
        print("epoch: ",epoch)
        checkpoint_path =  checkpoint + 'checkpoint-'+ str(epoch) +'.ckpt'
        video_infos = get_video_info(config['dataset']['testing']['csi_info_path'])
        originidx_to_idx, idx_to_class = get_class_index_map()

        npy_data_path = config['dataset']['testing']['csi_data_path']

        net = wifitad(in_channels=config['model']['in_channels'])
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()
        #os.remove(checkpoint_path)
        score_func = nn.Softmax(dim=-1)

        result_dict = {}
        for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
            sample_count = video_infos[video_name]['sample_count']
            sample_fps = video_infos[video_name]['sample_fps']
            if sample_count < clip_length:
                offsetlist = [0]
            else:
                offsetlist = list(range(0, sample_count - clip_length + 1, stride))
                if (sample_count - clip_length) % stride:
                    offsetlist += [sample_count - clip_length]

            data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
            data = np.transpose(data)
            data = torch.from_numpy(data)

            output = []
            for cl in range(num_classes):
                output.append([])
            res = torch.zeros(num_classes, top_k, 3)

            for offset in offsetlist:
                clip = data[:, offset: offset + clip_length]
                clip = clip.float()
                clip = clip / 40.0
                if clip.size(1) < clip_length:
                    tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                                    96, 96]).float()
                    clip = torch.cat([clip, tmp], dim=1)
                clip = clip.unsqueeze(0).cuda()
                with torch.no_grad():
                    output_dict = net(clip)
                loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors'][0]
                decoded_segments = torch.cat(
                    [priors[:, :1] * clip_length - loc[:, :1],
                    priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
                decoded_segments.clamp_(min=0, max=clip_length)

                conf = score_func(conf)
                conf = conf.view(-1, num_classes).transpose(1, 0)
                conf_scores = conf.clone()

                for cl in range(1, num_classes):
                    c_mask = conf_scores[cl] > conf_thresh
                    scores = conf_scores[cl][c_mask]
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                    segments = decoded_segments[l_mask].view(-1, 2)
                    segments = (segments + offset) / sample_fps
                    segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                    output[cl].append(segments)
            sum_count = 0
            for cl in range(1, num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)
                tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
                res[cl, :count] = tmp
                sum_count += count

            sum_count = min(sum_count, top_k)
            flt = res.contiguous().view(-1, 3)
            flt = flt.view(num_classes, -1, 3)
            proposal_list = []
            for cl in range(1, num_classes):
                class_name = idx_to_class[cl]
                tmp = flt[cl].contiguous()
                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {}
                    tmp_proposal['label'] = class_name
                    tmp_proposal['score'] = float(tmp[i, 2])
                    tmp_proposal['segment'] = [float(tmp[i, 0]),
                                            float(tmp[i, 1])]
                    proposal_list.append(tmp_proposal)

            result_dict[video_name] = proposal_list

        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
        json_name = "checkpoint" + str(epoch) + ".json"
        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)
