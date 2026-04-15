import torch
import argparse
import os
import sys
import math
import numpy as np
import yaml
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.yolo import YOLOV1
from tqdm import tqdm
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from loss.yolov1_loss import YOLOV1Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def collate_function(data):
    return list(zip(*data))


def _resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _resolve_dataset_path(path):
    resolved_path = _resolve_project_path(path)
    if os.path.exists(resolved_path):
        return resolved_path

    alt_path = os.path.join(PROJECT_ROOT, 'data', 'VOCdevkit', os.path.basename(path))
    if os.path.exists(alt_path):
        return alt_path
    return resolved_path


def _adjust_batch_size_for_device(train_config):
    if device.type != 'cuda':
        return

    total_memory_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    original_batch_size = train_config['batch_size']
    adjusted_batch_size = original_batch_size

    if total_memory_gib <= 8 and original_batch_size > 2:
        adjusted_batch_size = 2
    elif total_memory_gib <= 12 and original_batch_size > 4:
        adjusted_batch_size = 4
    elif total_memory_gib <= 16 and original_batch_size > 8:
        adjusted_batch_size = 8

    if adjusted_batch_size == original_batch_size:
        return

    scale = math.ceil(original_batch_size / adjusted_batch_size)
    train_config['batch_size'] = adjusted_batch_size
    train_config['acc_steps'] *= scale
    print(
        'Adjusted batch_size from {} to {} for {:.2f} GiB GPU memory; '
        'updated acc_steps to {}'.format(
            original_batch_size,
            adjusted_batch_size,
            total_memory_gib,
            train_config['acc_steps'],
        )
    )


def train(args):
    # Read the config file #
    config_path = _resolve_project_path(args.config_path)
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f'Failed to parse config file: {config_path}') from exc
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    dataset_config['train_im_sets'] = [
        _resolve_dataset_path(im_set)
        for im_set in dataset_config['train_im_sets']
    ]
    dataset_config['test_im_sets'] = [
        _resolve_dataset_path(im_set)
        for im_set in dataset_config['test_im_sets']
    ]
    train_config['task_name'] = _resolve_project_path(train_config['task_name'])
    _adjust_batch_size_for_device(train_config)

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'],
                     S=model_config['S'],
                     B=model_config['B'],
                     C=dataset_config['num_classes'])
    train_dataset = DataLoader(voc,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)

    yolo_model = YOLOV1(im_size=dataset_config['im_size'],
                        num_classes=dataset_config['num_classes'],
                        model_config=model_config)
    yolo_model.train()
    yolo_model.to(device)
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ckpt_name'])):
        print('Loading checkpoint as one exists')
        yolo_model.load_state_dict(torch.load(
            os.path.join(train_config['task_name'],
                         train_config['ckpt_name']),
            map_location=device))
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              yolo_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)

    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
    criterion = YOLOV1Loss()
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    for epoch_idx in range(num_epochs):
        losses = []
        optimizer.zero_grad()
        for idx, (ims, targets, _) in enumerate(
            tqdm(train_dataset, dynamic_ncols=True, bar_format=progress_bar_format)
        ):
            yolo_targets = torch.cat([
                target['yolo_targets'].unsqueeze(0).float().to(device)
                for target in targets], dim=0)
            im = torch.cat([im.unsqueeze(0).float().to(device) for im in ims], dim=0)
            yolo_preds = yolo_model(im)
            loss = criterion(yolo_preds, yolo_targets, use_sigmoid=model_config['use_sigmoid'])
            loss = loss / acc_steps
            loss.backward()
            losses.append(loss.item())
            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if steps % train_config['log_steps'] == 0:
                tqdm.write('Loss : {:.6f}'.format(np.mean(losses)))
            if torch.isnan(loss):
                tqdm.write('Loss is becoming nan. Exiting')
                exit(0)
            steps += 1
        tqdm.write('Finished epoch {}'.format(epoch_idx + 1))
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        torch.save(yolo_model.state_dict(), os.path.join(train_config['task_name'],
                                                         train_config['ckpt_name']))
    tqdm.write('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for yolov1 training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)
