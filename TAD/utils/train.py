# bash TAD/train_tools/tools.sh 3,1
import os
import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from TAD.dataset.load_csi import SmartWiFi, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from TAD.model.tad_model import wifitad
from TAD.losses.loss import MultiSegmentLoss
from TAD.config import config
from TAD.utils.device import get_device

batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']

device = get_device()
ngpu = (
    config.get('ngpu', torch.cuda.device_count())
    if device.type == "cuda"
    else 0
)
GLOBAL_SEED = 1

train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']

def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)
    print('gpu num: ', ngpu)


def set_seed(seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        # seed all CUDA devices
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # make cuDNN deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def get_rng_states():
    """Capture and return RNG states for Python, NumPy, and Torch (and CUDA if used)."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state()
    }
    if device.type == "cuda":
        states["torch_cuda"] = torch.cuda.get_rng_state()
    return states


def set_rng_states(states):
    """Restore RNG states from get_rng_states()."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in states:
        torch.cuda.set_rng_state(states["torch_cuda"])


def save_model(epoch, model, optimizer):
    # if model is DataParallel, unwrap it, otherwise use it directly
    base_model = model.module if hasattr(model, 'module') else model

    torch.save(
        base_model.state_dict(),
        os.path.join(checkpoint_path, f'checkpoint-{epoch}.ckpt')
    )
    # torch.save({'optimizer': optimizer.state_dict(),
    #             'state': get_rng_states()},
    #            os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(epoch)))

def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_states(state_dict['state'])
    return start_epoch

def forward_one_epoch(net, clips, targets, loss_fn, training=True):
    # send inputs to the same device as the model
    clips = clips.to(device)
    targets = [t.to(device) for t in targets]
    if training:
        output_dict = net(clips)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    preds = [output_dict['loc'], output_dict['conf'], output_dict["priors"][0]]
    loss_l, loss_c = loss_fn(preds, targets)
    return loss_l, loss_c

def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, loss_fn, training=True):
    if training:
        net.train()
    else:
        net.eval()

    iteration = 0
    loss_loc_val = 0
    loss_conf_val = 0
    cost_val = 0

    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets) in enumerate(pbar):
            iteration = n_iter
            loss_l, loss_c= forward_one_epoch(net, clips, targets, loss_fn, training=training)

            loss_l = loss_l * config['training']['lw'] * 100
            loss_c = loss_c * config['training']['cw']
            cost = loss_l + loss_c
            if training:
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

            loss_loc_val += loss_l.cpu().detach().numpy()
            loss_conf_val += loss_c.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    loss_loc_val /= (iteration + 1)
    loss_conf_val /= (iteration + 1)
    cost_val /= (iteration + 1)

    if training:
        prefix = 'Train'
        save_model(epoch, net, optimizer)
    else:
        prefix = 'Val'


    plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}'\
        .format(epoch, prefix, cost_val, loss_loc_val, loss_conf_val)
    print(plog)


def build_model():
    """
    Create model
    """
    net = wifitad(in_channels=config['model']['in_channels'])
    # move model to device
    net = net.to(device)
    # wrap in DataParallel only for CUDA + multiple GPUs
    if device.type == "cuda" and ngpu > 1:
        net = nn.DataParallel(net, device_ids=list(range(ngpu)))
    return net

def train():
    """
    Perform training
    """
    # Setup model
    net = build_model()

    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    
    # Setup loss
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)

    # Setup dataloader
    # Memory pinning only makes sense for CUDA
    use_pin_memory = (device.type == "cuda")
    train_video_infos = get_video_info(config['dataset']['training']['csi_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['csi_anno_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['csi_data_path'])
    train_dataset = SmartWiFi(train_data_dict,
                                   train_video_infos,
                                   train_video_annos)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        collate_fn=detection_collate,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    epoch_step_num = len(train_dataset) // batch_size

    # Start training
    start_epoch = resume_training(resume, net, optimizer)
    
    for i in range(start_epoch, max_epoch + 1):
        run_one_epoch(i, net, optimizer, train_data_loader, epoch_step_num, loss_fn=CPD_Loss)

if __name__ == '__main__':
    # NOTE: On mac and probably windows, multiprocessing doesnt fork.
    # This causes dangling processes which deadlock on cleanup, freezing
    # this script. This fixes such issues and should not impact the linux default.
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        # if it’s already set, just move on
        pass

    print_training_info()
    set_seed(random_seed)
    train()

    print("Training finished.")
