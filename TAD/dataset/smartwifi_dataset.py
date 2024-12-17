import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
import tqdm
from TAD.config import config
import random
import math

def get_class_index_map(class_info_path='dataset/annotations/Class Index_Detection.txt'):
    txt = np.loadtxt(class_info_path, dtype=str)
    originidx_to_idx = {}
    idx_to_class = {}
    for idx, l in enumerate(txt):
        originidx_to_idx[int(l[0])] = idx + 1
        idx_to_class[idx + 1] = l[1]
    return originidx_to_idx, idx_to_class


def get_video_info(video_info_path):
    df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            'fps': info[1],
            'sample_fps': info[2],
            'count': info[3],
            'sample_count': info[4]
        }
    return video_infos


def get_video_anno(video_infos,
                   video_anno_path):
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    originidx_to_idx, idx_to_class = get_class_index_map()
    video_annos = {}
    for anno in df_anno:
        video_name = anno[0]
        originidx = anno[2]
        start_frame = anno[-2]
        end_frame = anno[-1]
        count = video_infos[video_name]['count']
        sample_count = video_infos[video_name]['sample_count']
        ratio = sample_count * 1.0 / count
        start_gt = start_frame * ratio
        end_gt = end_frame * ratio
        class_idx = originidx_to_idx[originidx]
        if video_annos.get(video_name) is None:
            video_annos[video_name] = [[start_gt, end_gt, class_idx]]
        else:
            video_annos[video_name].append([start_gt, end_gt, class_idx])
    return video_annos


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2]
        ])
    return res


def split_videos(video_infos,
                 video_annos,
                 clip_length=config['dataset']['training']['clip_length'],
                 stride=config['dataset']['training']['clip_stride']):
    training_list = []
    for video_name in video_annos.keys():
        min_anno = clip_length
        sample_count = video_infos[video_name]['sample_count']
        annos = video_annos[video_name]
        if sample_count <= clip_length:
            offsetlist = [0]
            min_anno_len = min([x[1] - x[0] for x in annos])
            if min_anno_len < min_anno:
                min_anno = min_anno_len
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]
        for offset in offsetlist:
            left, right = offset + 1, offset + clip_length
            cur_annos = []
            save_offset = False
            for anno in annos:
                max_l = max(left, anno[0])
                min_r = min(right, anno[1])
                ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
                if ioa >= 1.0:
                    save_offset = True
                if ioa >= 0.5:
                    cur_annos.append([max(anno[0] - offset, 1),
                                      min(anno[1] - offset, clip_length),
                                      anno[2]])
            if save_offset:
                training_list.append({
                    'video_name': video_name,
                    'offset': offset,
                    'annos': cur_annos
                })
    return training_list


def load_video_data(video_infos, npy_data_path):
    data_dict = {}
    print('loading csi data ...')
    
    
    # 1D Conv changed original to 1, 8500, 30, 30, now is 1, 8500, 30, 1
    # ===========================================================================
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        data = np.transpose(data)
        data_dict[video_name] = data
    return data_dict


class SmartWiFi(Dataset):
    def __init__(self, data_dict,
                 video_infos,
                 video_annos,
                 clip_length=config['dataset']['training']['clip_length'],
                 stride=config['dataset']['training']['clip_stride'],
                 training=True,
                 origin_ratio=1):
        
        training_list = split_videos(
            video_infos,
            video_annos,
            clip_length,
            stride
        )
        self.training_list = training_list
        self.data_dict = data_dict
        self.clip_length = clip_length
        self.training = training
        self.origin_ratio = origin_ratio

    def __len__(self):
        return len(self.training_list)

    def get_bg(self, annos, min_action):
        annos = [[anno[0], anno[1]] for anno in annos]
        times = []
        for anno in annos:
            times.extend(anno)
        times.extend([0, self.clip_length - 1])
        times.sort()
        regions = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
        regions = list(filter(
            lambda x: x not in annos and math.floor(x[1]) - math.ceil(x[0]) > min_action, regions))
        region = random.choice(regions)
        return [math.ceil(region[0]), math.floor(region[1])]

    def augment_(self, input, annos, th):
        '''
        input: (c, t, h, w)
        target: (N, 3)
        '''
        try:
            gt = random.choice(list(filter(lambda x: x[1] - x[0] > 2 * th, annos)))
        except IndexError:
            return input, annos, False
        gt_len = gt[1] - gt[0]
        region = range(math.floor(th), math.ceil(gt_len - th))
        t = random.choice(region) + math.ceil(gt[0])
        try:
            bg = self.get_bg(annos, th)
        except IndexError:
            return input, annos, False
        start_idx = random.choice(range(bg[1] - bg[0] - th)) + bg[0]
        end_idx = start_idx + th

        new_input = input.clone()
        if gt[1] < start_idx:
            new_input[:, t:t + th, ] = input[:, start_idx:end_idx, ]
            new_input[:, t + th:end_idx, ] = input[:, t:start_idx, ]

            new_annos = [[gt[0], t], [t + th, th + gt[1]], [t + 1, t + th - 1]]
        else:
            new_input[:, start_idx:t - th] = input[:, end_idx:t, ]
            new_input[:, t - th:t, ] = input[:, start_idx:end_idx, ]

            new_annos = [[gt[0] - th, t - th], [t, gt[1]], [t - th + 1, t - 1]]

        return new_input, new_annos, True

    def augment(self, input, annos, th, max_iter=10):
        flag = True
        i = 0
        while flag and i < max_iter:
            new_input, new_annos, flag = self.augment_(input, annos, th)
            i += 1
        return new_input, new_annos, flag

    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        video_data = self.data_dict[sample_info['video_name']]
        offset = sample_info['offset']
        annos = sample_info['annos']

        input_data = video_data[:, offset: offset + self.clip_length]
        input_data = torch.from_numpy(input_data).float()
        # 数据范围在0-40
        # input_data = (input_data / 40.0)* 2.0 - 1.0
        input_data = input_data / 40.0
        annos = annos_transform(annos, self.clip_length)
        target = np.stack(annos, 0)
        return input_data, target


def detection_collate(batch):
    clips = []
    targets = []
    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(clips, 0), targets
